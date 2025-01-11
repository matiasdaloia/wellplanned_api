# WellPlanned AI

## Setup

1. Create a Supabase project at https://supabase.com

2. Create a storage bucket:
   - Go to Storage in your Supabase dashboard
   - Create a new bucket named "pdfs"
   - Set the bucket's privacy to "private"

3. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then add your values:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_KEY`: Your Supabase anon/public key
   - Other required API keys

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run database migrations:
   ```bash
   # Install Supabase CLI
   brew install supabase/tap/supabase

   # Login to Supabase
   supabase login

   # Link your project
   supabase link --project-ref your-project-ref

   # Run migrations
   supabase db push
   ```

6. Run the development server:
   ```bash
   uvicorn main:app --reload
   ```

## API Documentation

### Authentication

All API endpoints require authentication using a JWT token from Supabase. The token should be included in the Authorization header of each request.

#### React Native Client Setup
```javascript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  'YOUR_SUPABASE_URL',
  'YOUR_SUPABASE_ANON_KEY'
)

// Sign up
const { data, error } = await supabase.auth.signUp({
  email: 'user@example.com',
  password: 'your-password',
  options: {
    data: {
      first_name: 'John',
      last_name: 'Doe'
    }
  }
})

// Sign in
const { data, error } = await supabase.auth.signInWithPassword({
  email: 'user@example.com',
  password: 'your-password'
})

// Get session token for API requests
const { data: { session } } = await supabase.auth.getSession()
const token = session?.access_token

// Use token in API requests
const response = await fetch('YOUR_API_URL/endpoint', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
})
```

#### API Endpoints

##### Sign Up
```http
POST /auth/signup
{
    "email": "user@example.com",
    "password": "your-password",
    "first_name": "John",
    "last_name": "Doe"
}
```

##### Sign In
```http
POST /auth/signin
{
    "email": "user@example.com",
    "password": "your-password"
}
```

##### Sign Out
```http
POST /auth/signout
Authorization: Bearer your-jwt-token
```

Note: The JWT token used in the Authorization header should be obtained from Supabase's `auth.getSession()` in your React Native client. This token is automatically handled by the Supabase client but needs to be manually included in requests to this API.

### Profile Management

#### Get Profile
```http
GET /profile
Authorization: Bearer your-jwt-token
```

#### Update Profile
```http
PUT /profile
Authorization: Bearer your-jwt-token
{
    "first_name": "John",
    "last_name": "Doe"
}
```

### Meal Plans

All meal plan endpoints require authentication with a JWT token in the Authorization header.

#### Create Meal Plan
```http
POST /mealplans
{
    "weekday": 0,
    "meals": [
        {
            "slot": 0,
            "meal": "Oatmeal with fruits",
            "ingredients": ["oats", "banana", "berries"],
            "recipe": {
                "title": "Healthy Breakfast Bowl",
                "ingredients": [
                    "1 cup rolled oats",
                    "1 banana, sliced",
                    "1/2 cup mixed berries"
                ],
                "steps": [
                    "Cook oats according to package instructions",
                    "Top with sliced banana and berries"
                ]
            }
        }
    ],
    "pdf_url": "optional-pdf-url",
    "data": {
        "additional": "metadata"
    }
}
```

#### Get Meal Plan
```http
GET /mealplans/{meal_plan_id}
```

#### Get Meal Plan with Recipes
```http
GET /mealplans/{meal_plan_id}/with-recipes
```

#### List Meal Plans
```http
GET /mealplans
```

#### Update Meal Plan
```http
PUT /mealplans/{meal_plan_id}
{
    "weekday": 0,
    "meals": [...]
}
```

#### Delete Meal Plan
```http
DELETE /mealplans/{meal_plan_id}
```

### Recipes

All recipe endpoints require authentication with a JWT token in the Authorization header. Recipes are always associated with a specific meal in a meal plan.

#### Create Recipe
```http
POST /recipes
{
    "recipe_url": "https://example.com/recipe",
    "language": "en"
}
```

Note: Recipes are typically created automatically when creating a meal plan with recipe information. This endpoint is mainly used for creating standalone recipes.

#### Get Recipe
```http
GET /recipes/{recipe_id}
```

#### List Recipes
```http
GET /recipes
```

#### Update Recipe
```http
PUT /recipes/{recipe_id}
{
    "recipe_url": "https://example.com/recipe",
    "language": "en"
}
```

#### Delete Recipe
```http
DELETE /recipes/{recipe_id}
```

## Database Schema

### Profiles
- `id`: UUID (references auth.users)
- `first_name`: Text
- `last_name`: Text
- `updated_at`: Timestamp

### Meal Plans
- `id`: UUID
- `profile_id`: UUID (references profiles)
- `pdf_url`: Text
- `data`: JSONB
- `created_at`: Timestamp
- `updated_at`: Timestamp

### Meal Plan Recipes
- `id`: UUID
- `meal_plan_id`: UUID (references meal_plans)
- `weekday`: Integer (0-6)
- `meal_slot`: Integer (0-4)
- `created_at`: Timestamp
- `updated_at`: Timestamp

### Recipes
- `id`: UUID
- `meal_plan_recipe_id`: UUID (references meal_plan_recipes)
- `profile_id`: UUID (references profiles)
- `title`: Text
- `thumbnail`: Text
- `author`: Text
- `difficulty`: Text
- `time`: Text
- `servings`: Text
- `ingredients`: JSONB
- `steps`: JSONB
- `created_at`: Timestamp
- `updated_at`: Timestamp

## Meal Slots

The `meal_slot` in meal plan recipes represents:
- 0: Breakfast
- 1: Mid Morning Snack
- 2: Lunch
- 3: Afternoon Snack
- 4: Dinner

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.
