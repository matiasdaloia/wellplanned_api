from fastapi import APIRouter, Depends, HTTPException

from src.routers.auth import get_current_user
from src.utils.supabase_client import supabase

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.get("/{recipe_id}")
async def get_recipe(recipe_id: str, user=Depends(get_current_user)):
    """Get a recipe by ID"""
    recipe = await supabase.get_recipe(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe


@router.get("")
async def list_recipes(user=Depends(get_current_user)):
    """List all recipes"""
    return await supabase.list_recipes()


@router.delete("/{recipe_id}")
async def delete_recipe(recipe_id: str, user=Depends(get_current_user)):
    """Delete a recipe"""
    await supabase.delete_recipe(recipe_id)
    return {"message": "Recipe deleted"}
