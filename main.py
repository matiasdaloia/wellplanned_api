from dotenv import load_dotenv
from fastapi import FastAPI

from src.routers import auth, meal_plans, profiles, recipes, recommendations
from src.utils.middleware import setup_middleware

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Set up middleware
setup_middleware(app)

# Include routers
app.include_router(auth.router)
app.include_router(profiles.router)
app.include_router(meal_plans.router)
app.include_router(recipes.router)
app.include_router(recommendations.router)
