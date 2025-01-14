# This file is part of WellPlanned AI.
#
# WellPlanned AI is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# WellPlanned AI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with WellPlanned AI. If not, see <https://www.gnu.org/licenses/>.

import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException

from supabase import Client, create_client

load_dotenv()


class SupabaseClient:
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "Supabase URL and key must be set in environment variables"
            )

        self.client: Client = create_client(supabase_url, supabase_key)

    # Auth and User Profile
    async def sign_up(
        self, email: str, password: str, first_name: str, last_name: str
    ) -> Dict[str, Any]:
        """Sign up a new user"""
        try:
            auth_response = await self.client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {
                        "data": {"first_name": first_name, "last_name": last_name}
                    },
                }
            )
            return auth_response
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in a user"""
        try:
            return await self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    async def sign_out(self) -> None:
        """Sign out the current user"""
        await self.client.auth.sign_out()

    def get_user_by_jwt(self, jwt: str) -> Optional[Dict[str, Any]]:
        """Get user from JWT token"""
        try:
            # Set the session token for this request
            self.client.auth.set_session(jwt, jwt)
            return self.client.auth.get_user(jwt)
        except Exception as e:
            print(f"Error getting user from JWT: {e}")
            return None

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's profile"""
        response = self.client.table("profiles").select("*").eq("id", user_id).execute()

        return response.data[0] if response.data else None

    def update_profile(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a user's profile"""
        return self.client.table("profiles").update(data).eq("id", user_id).execute()

    # Meal Plans
    def create_meal_plan(self, profile_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new meal plan"""
        meal_plan_data = {
            "profile_id": profile_id,
            "pdf_url": data.get("pdf_url"),
            "data": data.get("data"),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        meal_plan = self.client.table("meal_plans").insert(meal_plan_data).execute()

        return meal_plan

    async def get_meal_plan(self, meal_plan_id: str) -> Optional[Dict[str, Any]]:
        """Get a meal plan by ID"""
        response = (
            self.client.table("meal_plans").select("*").eq("id", meal_plan_id).execute()
        )
        return response.data[0] if response.data else None

    def list_meal_plans(self) -> List[Dict[str, Any]]:
        """List all meal plans sorted by creation date (newest first)"""
        response = (
            self.client.table("meal_plans")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data

    async def update_meal_plan(
        self, meal_plan_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a meal plan"""
        return (
            self.client.table("meal_plans")
            .update(data)
            .eq("id", meal_plan_id)
            .execute()
        )

    async def delete_meal_plan(self, meal_plan_id: str) -> None:
        """Delete a meal plan"""
        self.client.table("meal_plans").delete().eq("id", meal_plan_id).execute()

    # Recipes
    async def create_recipe(
        self, profile_id: str, meal_plan_recipe_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new recipe"""
        recipe_data = {
            "profile_id": profile_id,
            "meal_plan_recipe_id": meal_plan_recipe_id,
            **data,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        return await self.client.table("recipes").insert(recipe_data).execute()

    async def get_meal_plan_with_recipes(
        self, meal_plan_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a meal plan with all its recipes"""
        meal_plan = await self.get_meal_plan(meal_plan_id)
        if not meal_plan:
            return None

        # Get all meal plan recipes
        meal_plan_recipes = (
            await self.client.table("meal_plan_recipes")
            .select("*")
            .eq("meal_plan_id", meal_plan_id)
            .execute()
        )

        # Get recipes for each meal plan recipe
        for mpr in meal_plan_recipes.data:
            recipe = (
                await self.client.table("recipes")
                .select("*")
                .eq("meal_plan_recipe_id", mpr["id"])
                .execute()
            )
            mpr["recipe"] = recipe.data[0] if recipe.data else None

        meal_plan["meal_plan_recipes"] = meal_plan_recipes.data
        return meal_plan

    async def get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """Get a recipe by ID"""
        response = (
            self.client.table("recipes").select("*").eq("id", recipe_id).execute()
        )
        return response.data[0] if response.data else None

    async def list_recipes(self) -> List[Dict[str, Any]]:
        """List all recipes"""
        response = self.client.table("recipes").select("*").execute()
        return response.data

    async def update_recipe(
        self, recipe_id: str, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a recipe"""
        return self.client.table("recipes").update(data).eq("id", recipe_id).execute()

    async def delete_recipe(self, recipe_id: str) -> None:
        """Delete a recipe"""
        self.client.table("recipes").delete().eq("id", recipe_id).execute()

    # File Storage
    def upload_file(
        self, bucket: str, file_path: str, file_data: bytes, file_options: dict
    ) -> str:
        """
        Upload a file to Supabase storage
        Returns the public URL of the uploaded file
        """
        self.client.storage.from_(bucket).upload(file_path, file_data, file_options)
        # Get public URL
        return self.client.storage.from_(bucket).get_public_url(file_path)

    async def delete_file(self, bucket: str, file_path: str) -> None:
        """Delete a file from storage"""
        self.client.storage.from_(bucket).remove([file_path])

    async def get_file_url(self, bucket: str, file_path: str) -> str:
        """
        Get the public URL of a file in storage
        Args:
            bucket: The name of the bucket
            file_path: The path to the file within the bucket
        Returns:
            The public URL of the file
        """
        return self.client.storage.from_(bucket).get_public_url(file_path)

    async def download_file(self, bucket: str, file_path: str) -> bytes:
        """
        Download a file from storage
        Args:
            bucket: The name of the bucket
            file_path: The path to the file within the bucket
        Returns:
            The file content as bytes
        """
        return self.client.storage.from_(bucket).download(file_path)

    async def list_files(self, bucket: str, path: str = None) -> List[Dict[str, Any]]:
        """
        List all files in a bucket
        Args:
            bucket: The name of the bucket
            path: Optional path within the bucket to list files from
        Returns:
            List of file objects
        """
        return self.client.storage.from_(bucket).list(path=path)

    async def save_recommendations(
        self, profile_id: str, meal_plan_id: str, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Save recipe recommendations to the database
        Args:
            profile_id: The ID of the user profile
            meal_plan_id: The ID of the meal plan
            recommendations: List of recommendation objects containing recipe details
        Returns:
            The saved recommendations
        """
        recommendations_data = [
            {
                "profile_id": profile_id,
                "meal_plan_id": meal_plan_id,
                "weekday": rec["weekday"],
                "slot": rec["slot"],
                "recipe_title": rec["recipe_title"],
                "recipe_link": rec["recipe_link"],
                "recipe_thumbnail": rec["recipe_thumbnail"],
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
            for rec in recommendations
        ]

        response = (
            self.client.table("recommendations").insert(recommendations_data).execute()
        )
        return response.data

    def get_recommendations(
        self, meal_plan_id: str, weekday: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for a specific meal plan
        Args:
            meal_plan_id: The ID of the meal plan
            weekday: Optional weekday to filter recommendations (0 for Monday, 6 for Sunday)
        Returns:
            List of recommendations for the meal plan
        """
        query = (
            self.client.table("recommendations")
            .select("*")
            .eq("meal_plan_id", meal_plan_id)
        )

        if weekday is not None:
            query = query.eq("weekday", weekday)

        response = query.order("created_at", desc=True).execute()
        return response.data

    async def update_recipe_breakdown(
        self, recommendation_id: str, breakdown: Dict[str, Any], status: str
    ) -> Dict[str, Any]:
        """
        Update recipe breakdown for a recommendation
        Args:
            recommendation_id: The ID of the recommendation
            breakdown: The recipe breakdown data
            status: The status of the breakdown generation (pending, completed, failed)
        Returns:
            The updated recommendation
        """
        data = {
            "recipe_breakdown_content": breakdown,
            "recipe_breakdown_status": status,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        response = (
            self.client.table("recommendations")
            .update(data)
            .eq("id", recommendation_id)
            .execute()
        )
        return response.data[0] if response.data else None

    async def get_recommendation(
        self, recommendation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a recommendation by ID
        Args:
            recommendation_id: The ID of the recommendation
        Returns:
            The recommendation data or None if not found
        """
        response = (
            self.client.table("recommendations")
            .select("*")
            .eq("id", recommendation_id)
            .execute()
        )
        return response.data[0] if response.data else None


# Create a singleton instance
supabase = SupabaseClient()
