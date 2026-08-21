-- Missing-aware O*NET 30.3 Transferable Skills + Knowledge ratings.
-- Missing, suppressed, and not-relevant observations retain provenance and
-- precision fields but never receive a normalized feature value.

CREATE OR REPLACE TABLE skill_ratings AS
WITH combined AS (
    SELECT 'transferable_skill'::VARCHAR AS feature_family, *
    FROM stg_transferable_skills
    UNION ALL BY NAME
    SELECT 'knowledge'::VARCHAR AS feature_family, *
    FROM stg_knowledge
),
typed AS (
    SELECT
        onet_soc_code::VARCHAR AS onet_soc_code,
        feature_family,
        trim(element_id)::VARCHAR AS element_id,
        trim(element_name)::VARCHAR AS element_name,
        upper(trim(scale_id))::VARCHAR AS scale_id,
        trim(scale_name)::VARCHAR AS scale_name,
        try_cast(data_value AS DOUBLE) AS raw_value,
        try_cast(n AS BIGINT) AS sample_size,
        try_cast(standard_error AS DOUBLE) AS standard_error,
        try_cast(lower_ci_bound AS DOUBLE) AS lower_ci_bound,
        try_cast(upper_ci_bound AS DOUBLE) AS upper_ci_bound,
        lower(coalesce(trim(recommend_suppress), '')) IN ('y', 'yes', 'true', '1') AS recommend_suppress,
        lower(coalesce(trim(not_relevant), '')) IN ('y', 'yes', 'true', '1') AS not_relevant,
        coalesce(
            try_cast(date AS DATE),
            try_strptime(date, '%m/%Y')::DATE,
            try_strptime(date, '%Y-%m')::DATE
        ) AS rating_date,
        nullif(regexp_replace(trim(domain_source), '\s+', ' ', 'g'), '')::VARCHAR AS domain_source
    FROM combined
)
SELECT
    *,
    raw_value IS NOT NULL AND NOT recommend_suppress AND NOT not_relevant AS observed,
    CASE
        WHEN raw_value IS NULL OR recommend_suppress OR not_relevant THEN NULL
        WHEN scale_id = 'IM' THEN (raw_value - 1.0) / 4.0
        WHEN scale_id = 'LV' THEN raw_value / 7.0
    END::DOUBLE AS normalized_value
FROM typed
ORDER BY onet_soc_code, feature_family, element_id, scale_id;

CREATE OR REPLACE VIEW structured_ratings AS
SELECT * FROM skill_ratings;
