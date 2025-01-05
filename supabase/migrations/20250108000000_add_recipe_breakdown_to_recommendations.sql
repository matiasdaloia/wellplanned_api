-- Add recipe breakdown columns to recommendations table
alter table
    recommendations
add
    column recipe_breakdown_content text,
add
    column recipe_breakdown_status text default 'pending' not null;

-- Add index for status queries
create index recommendations_recipe_breakdown_status_idx on recommendations(recipe_breakdown_status);