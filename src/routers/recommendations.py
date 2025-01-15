import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.routers.auth import get_current_user
from src.utils.logger import get_logger, log_execution_time_async
from src.utils.supabase_client import supabase

logger = get_logger()
router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class RecipeBreakdown(TypedDict):
    title: str
    author: str
    difficulty: str
    time: str
    servings: str
    ingredients: List[str]
    steps: List[str]


def get_current_weekday() -> int:
    """Get the current weekday (0 for Monday, 6 for Sunday)"""
    return datetime.now().weekday()


async def stream_recommendations(latest_meal_plan: Dict[str, Any], target_weekday: int):
    try:
        google_search = GoogleSerperAPIWrapper(
            serper_api_key=os.getenv("SERPER_API_KEY"), type="images"
        )
        recommendations = []

        # Filter the meal plan to only include the target weekday
        current_day_meals = [
            day for day in latest_meal_plan if day["weekday"] == target_weekday
        ]

        for day in current_day_meals:
            for meal in day["meals"]:
                search_result = await google_search.aresults(meal["recipeQuery"])

                top_3_results = search_result["images"][:3]

                meal_recommendations = []
                for result in top_3_results:
                    recommendation = {
                        "recipe_title": result["title"],
                        "recipe_link": result["link"],
                        "recipe_thumbnail": result["imageUrl"],
                        "weekday": day["weekday"],
                        "slot": meal["slot"],
                    }
                    meal_recommendations.append(recommendation)
                    recommendations.append(recommendation)

                yield f"data: {json.dumps({'type': 'update', 'content': meal_recommendations})}\n\n"

        yield f"data: {json.dumps({'type': 'complete', 'content': recommendations})}\n\n"

    except Exception as e:
        logger.error("Error streaming recommendations", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


async def stream_recipe_breakdown(url: str, language: str = "en"):
    """Stream recipe breakdown generation in structured format"""
    try:
        # Initialize the loader with a longer timeout and retry mechanism
        loader = SeleniumURLLoader(
            urls=[url],
        )

        documents = loader.load()

        combined_content = "\n\n".join(doc.page_content for doc in documents)

        # Use OpenAI for structured recipe parsing
        llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4o",
            stream_usage=True,
        )
        structured_llm = llm.with_structured_output(RecipeBreakdown)

        messages = [
            SystemMessage(
                content="""
            Analyze the recipe webpage content and extract the recipe information in a structured format.

            Extract the following information:
            - title: The title of the recipe
            - author: The author of the recipe (if available)
            - difficulty: The difficulty level (if available)
            - time: The time it takes to prepare the recipe (if available)
            - servings: The number of servings (use 4 if not specified, or adjust ingredients for 4 servings if more)
            - ingredients: List of ingredients with quantities
            - steps: List of preparation steps

            Only include factual information from the webpage. If a field is not available, omit it.
            Do not make up or infer missing information."""
            ),
            HumanMessage(content=combined_content),
        ]

        # Initialize an empty response
        full_response = None
        async for chunk in structured_llm.astream(messages):
            logger.debug(
                "Received chunk from LLM", extra={"extra_data": {"chunk": str(chunk)}}
            )
            full_response = chunk
            yield f"data: {json.dumps({'type': 'update', 'content': chunk})}\n\n"

        # Send the final complete response
        yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"

    except Exception as e:
        logger.error("Error generating recipe breakdown", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


@router.post("/{recommendation_id}/breakdown")
async def generate_recipe_breakdown(
    recommendation_id: str, user=Depends(get_current_user)
):
    """Generate recipe breakdown for a recommendation"""
    try:
        # Get recommendation
        recommendation = await supabase.get_recommendation(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")

        # Create async generator that will both stream the response and save the breakdown
        async def generate_and_save():
            full_response = None
            async for chunk in stream_recipe_breakdown(recommendation["recipe_link"]):
                # Parse the chunk to get the breakdown data
                chunk_data = json.loads(chunk.replace("data: ", ""))

                # If this is the complete response, save the markdown breakdown
                if chunk_data["type"] == "complete":
                    structured_data = chunk_data["content"]
                    # Save the structured breakdown in the background
                    asyncio.create_task(
                        supabase.update_recipe_breakdown(
                            recommendation_id, structured_data, "completed"
                        )
                    )
                elif chunk_data["type"] == "error":
                    # Update status to failed if there's an error
                    asyncio.create_task(
                        supabase.update_recipe_breakdown(
                            recommendation_id, None, "failed"
                        )
                    )

                yield chunk

        return StreamingResponse(
            generate_and_save(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error("Error generating recipe breakdown", exc_info=True)
        # Update status to failed
        await supabase.update_recipe_breakdown(recommendation_id, None, "failed")
        raise HTTPException(
            status_code=500, detail=f"Error generating recipe breakdown: {str(e)}"
        )


@router.get("/{recommendation_id}")
async def get_recommendation(recommendation_id: str, user=Depends(get_current_user)):
    """Get a recommendation by ID including its recipe breakdown"""
    recommendation = await supabase.get_recommendation(recommendation_id)
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return recommendation


@router.post("/generate")
@log_execution_time_async
async def generate_mealplan_recommendations(
    weekday: Optional[int] = None,
    user=Depends(get_current_user),
):
    try:
        # Fetch the latest meal plan for the user
        meal_plans = supabase.list_meal_plans()
        if not meal_plans:
            raise HTTPException(
                status_code=404, detail="No meal plans found for the user."
            )

        meal_plan = meal_plans[0]  # Get the full meal plan object
        latest_meal_plan = meal_plan.get("data", {}).get(
            "results", []
        )  # Access the data field which contains the meal plan structure

        target_weekday = weekday if weekday is not None else get_current_weekday()

        # Create an async generator that will both stream the response and save the recommendations
        async def generate_and_save():
            full_recommendations = None
            saved_recommendations = []
            async for chunk in stream_recommendations(latest_meal_plan, target_weekday):
                # Parse the chunk to get the recommendations data
                chunk_data = json.loads(chunk.replace("data: ", ""))

                # If this is the complete response, save the recommendations
                if chunk_data["type"] == "complete":
                    full_recommendations = chunk_data["content"]
                    # Save the recommendations and get their IDs
                    saved_recommendations = await supabase.save_recommendations(
                        user["id"], meal_plan["id"], full_recommendations
                    )
                    # Update the response to include the saved recommendation IDs
                    chunk_data["content"] = saved_recommendations
                    chunk = f"data: {json.dumps(chunk_data)}\n\n"

                yield chunk

        return StreamingResponse(
            generate_and_save(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error("Error generating meal plan recommendations", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating recommendations: {str(e)}"
        )


@router.get("/meal-plan/{meal_plan_id}")
async def get_meal_plan_recommendations(
    meal_plan_id: str, weekday: Optional[int] = None, user=Depends(get_current_user)
):
    """Get recommendations for a specific meal plan for the specified weekday (defaults to current weekday)"""
    target_weekday = weekday if weekday is not None else get_current_weekday()
    recommendations = supabase.get_recommendations(meal_plan_id, target_weekday)
    if not recommendations:
        raise HTTPException(
            status_code=404, detail="No recommendations found for this meal plan"
        )
    return recommendations
