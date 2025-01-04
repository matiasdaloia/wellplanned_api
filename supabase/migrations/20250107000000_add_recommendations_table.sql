-- Create recommendations table
create table recommendations (
    id uuid default uuid_generate_v4() primary key,
    profile_id uuid references profiles(id) on delete cascade not null,
    meal_plan_id uuid references meal_plans(id) on delete cascade not null,
    weekday int not null,
    slot int not null,
    recipe_title text not null,
    recipe_link text not null,
    recipe_thumbnail text not null,
    created_at timestamp with time zone default timezone('utc' :: text, now()) not null,
    updated_at timestamp with time zone default timezone('utc' :: text, now()) not null
);

-- Add indexes
create index recommendations_profile_id_idx on recommendations(profile_id);

create index recommendations_meal_plan_id_idx on recommendations(meal_plan_id);