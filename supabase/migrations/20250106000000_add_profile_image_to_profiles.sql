-- Add profile_image column to profiles table
alter table
    public.profiles
add
    column profile_image text,
add
    column language text;