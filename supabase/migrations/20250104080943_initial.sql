-- Create profiles table
create table public.profiles (
    id uuid references auth.users on delete cascade,
    first_name text,
    last_name text,
    updated_at timestamp with time zone,
    primary key (id)
);

-- Create meal_plans table
create table public.meal_plans (
    id uuid default uuid_generate_v4() primary key,
    profile_id uuid references public.profiles on delete cascade not null,
    pdf_url text,
    created_at timestamp with time zone default timezone('utc' :: text, now()) not null,
    updated_at timestamp with time zone default timezone('utc' :: text, now()) not null,
    data jsonb not null
);

-- Create meal_plan_recipes table to store recipes for specific meal slots
create table public.meal_plan_recipes (
    id uuid default uuid_generate_v4() primary key,
    meal_plan_id uuid references public.meal_plans on delete cascade not null,
    weekday integer not null check (
        weekday >= 0
        and weekday <= 6
    ),
    meal_slot integer not null check (
        meal_slot >= 0
        and meal_slot <= 4
    ),
    created_at timestamp with time zone default timezone('utc' :: text, now()) not null,
    updated_at timestamp with time zone default timezone('utc' :: text, now()) not null
);

-- Create recipes table
create table public.recipes (
    id uuid default uuid_generate_v4() primary key,
    meal_plan_recipe_id uuid references public.meal_plan_recipes on delete cascade not null,
    profile_id uuid references public.profiles on delete cascade not null,
    title text not null,
    thumbnail text,
    author text,
    difficulty text,
    time text,
    servings text,
    ingredients jsonb,
    steps jsonb,
    created_at timestamp with time zone default timezone('utc' :: text, now()) not null,
    updated_at timestamp with time zone default timezone('utc' :: text, now()) not null
);

-- Enable RLS
alter table
    public.profiles enable row level security;

alter table
    public.meal_plans enable row level security;

alter table
    public.recipes enable row level security;

-- Create policies
create policy "Users can view their own profile" on public.profiles for
select
    using (auth.uid() = id);

create policy "Users can update their own profile" on public.profiles for
update
    using (auth.uid() = id);

-- Enable RLS on new table
alter table
    public.meal_plan_recipes enable row level security;

-- Create policies for meal plans
create policy "Users can view their own meal plans" on public.meal_plans for
select
    using (auth.uid() = profile_id);

create policy "Users can create their own meal plans" on public.meal_plans for
insert
    with check (auth.uid() = profile_id);

create policy "Users can update their own meal plans" on public.meal_plans for
update
    using (auth.uid() = profile_id);

create policy "Users can delete their own meal plans" on public.meal_plans for delete using (auth.uid() = profile_id);

-- Create policies for meal plan recipes
create policy "Users can view their meal plan recipes" on public.meal_plan_recipes for
select
    using (
        exists (
            select
                1
            from
                public.meal_plans
            where
                id = meal_plan_id
                and profile_id = auth.uid()
        )
    );

create policy "Users can create their meal plan recipes" on public.meal_plan_recipes for
insert
    with check (
        exists (
            select
                1
            from
                public.meal_plans
            where
                id = meal_plan_id
                and profile_id = auth.uid()
        )
    );

create policy "Users can update their meal plan recipes" on public.meal_plan_recipes for
update
    using (
        exists (
            select
                1
            from
                public.meal_plans
            where
                id = meal_plan_id
                and profile_id = auth.uid()
        )
    );

create policy "Users can delete their meal plan recipes" on public.meal_plan_recipes for delete using (
    exists (
        select
            1
        from
            public.meal_plans
        where
            id = meal_plan_id
            and profile_id = auth.uid()
    )
);

-- Create policies for recipes
create policy "Users can view their own recipes" on public.recipes for
select
    using (auth.uid() = profile_id);

create policy "Users can create their own recipes" on public.recipes for
insert
    with check (auth.uid() = profile_id);

create policy "Users can update their own recipes" on public.recipes for
update
    using (auth.uid() = profile_id);

create policy "Users can delete their own recipes" on public.recipes for delete using (auth.uid() = profile_id);

-- Create profile on user signup
create function public.handle_new_user() returns trigger language plpgsql security definer
set
    search_path = public as $$ begin
insert into
    public.profiles (id, first_name, last_name, updated_at)
values
    (
        new.id,
        new.raw_user_meta_data ->> 'first_name',
        new.raw_user_meta_data ->> 'last_name',
        now()
    );

return new;

end;

$$;

-- Trigger for creating profile
create trigger on_auth_user_created
after
insert
    on auth.users for each row execute procedure public.handle_new_user();

-- Function to automatically update updated_at
create function public.handle_updated_at() returns trigger language plpgsql as $$ begin new.updated_at = now();

return new;

end;

$$;

-- Add updated_at triggers
create trigger handle_updated_at_profiles before
update
    on public.profiles for each row execute procedure public.handle_updated_at();

create trigger handle_updated_at_meal_plans before
update
    on public.meal_plans for each row execute procedure public.handle_updated_at();

create trigger handle_updated_at_meal_plan_recipes before
update
    on public.meal_plan_recipes for each row execute procedure public.handle_updated_at();

create trigger handle_updated_at_recipes before
update
    on public.recipes for each row execute procedure public.handle_updated_at();