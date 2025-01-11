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
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with WellPlanned AI. If not, see <https://www.gnu.org/licenses/>.

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
