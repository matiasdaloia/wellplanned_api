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

-- Add profile_image column to profiles table
alter table
    public.profiles
add
    column profile_image text,
add
    column language text;
