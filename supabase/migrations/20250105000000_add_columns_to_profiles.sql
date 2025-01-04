-- Add new columns to profiles table
alter table
    public.profiles
add
    column is_onboarded boolean,
add
    column allergies text [],
add
    column sports text [],
add
    column country text,
add
    column date_of_birth date,
add
    column sports_time_per_week integer,
add
    column diet_restrictions text [];