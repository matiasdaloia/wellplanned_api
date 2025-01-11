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

#### Generate Meal Plan Overview
```http
POST /mealplans/generate/overview
Authorization: Bearer your-jwt-token
```

#### Generate Meal Plan Recommendations
```http
POST /mealplans/generate/recommendations
Authorization: Bearer your-jwt-token
```

#### Check if Meal Plan Exists
```http
GET /mealplans/exists
Authorization: Bearer your-jwt-token
```

### Recipes

All recipe endpoints require authentication with a JWT token in the Authorization header. Recipes are always associated with a specific meal in a meal plan.

#### Generate Recipe Breakdown
```http
POST /recommendations/{recommendation_id}/breakdown
Authorization: Bearer your-jwt-token
```

## Contributing

We welcome contributions from the community! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## Changelog

All notable changes to this project will be documented in the [CHANGELOG.md](CHANGELOG.md) file.

## License

This project is licensed under the terms of the GNU Affero General Public License v3.0. See the [LICENSE](LICENSE) file for details.
