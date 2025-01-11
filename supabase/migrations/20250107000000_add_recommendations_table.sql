-- This file is part of WellPlanned AI.
--
-- WellPlanned AI is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- WellPlanned AI is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
-- GNU Affero General Public License for more details.
--
-- You should have received a copy of the GNU Affero General Public License
-- along with WellPlanned AI. If not, see <https://www.gnu.org/licenses/>.

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
