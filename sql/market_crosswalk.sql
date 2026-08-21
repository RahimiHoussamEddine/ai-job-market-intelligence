-- BLS Employment Projections 2024-34 Table 1.2 and official NEM/O*NET bridge.
-- Staging column names include explicit `_thousands` unit suffixes.  Summary
-- rows are excluded and no measure is ever summed across O*NET specialties.

CREATE OR REPLACE TABLE bls_market AS
SELECT
    trim(nem_code)::VARCHAR AS nem_code,
    regexp_replace(trim(nem_title), '\s+', ' ', 'g')::VARCHAR AS nem_title,
    trim(occupation_type)::VARCHAR AS occupation_type,
    round(try_cast(employment_2024_thousands AS DECIMAL(18, 6)) * 1000)::BIGINT AS employment_2024,
    round(try_cast(employment_2034_thousands AS DECIMAL(18, 6)) * 1000)::BIGINT AS employment_2034,
    round(try_cast(employment_change_2024_34_thousands AS DECIMAL(18, 6)) * 1000)::BIGINT AS employment_change_2024_34,
    try_cast(replace(employment_change_pct, '%', '') AS DOUBLE) AS employment_change_pct,
    round(try_cast(annual_openings_2024_34_thousands AS DECIMAL(18, 6)) * 1000)::BIGINT AS annual_openings_2024_34,
    try_cast(
        regexp_replace(median_annual_wage_2024, '[^0-9.-]', '', 'g')
        AS BIGINT
    ) AS median_annual_wage_2024,
    nullif(regexp_replace(trim(typical_education), '\s+', ' ', 'g'), '')::VARCHAR AS typical_education,
    nullif(regexp_replace(trim(work_experience), '\s+', ' ', 'g'), '')::VARCHAR AS work_experience,
    nullif(regexp_replace(trim(on_the_job_training), '\s+', ' ', 'g'), '')::VARCHAR AS on_the_job_training
FROM stg_bls_occupation
WHERE lower(trim(occupation_type)) = 'line item'
ORDER BY nem_code;

CREATE OR REPLACE TABLE onet_bls_crosswalk AS
SELECT
    trim(onet_soc_code)::VARCHAR AS onet_soc_code,
    nullif(regexp_replace(trim(onet_soc_title), '\s+', ' ', 'g'), '')::VARCHAR AS onet_soc_title,
    trim(nem_code)::VARCHAR AS nem_code,
    nullif(regexp_replace(trim(nem_title), '\s+', ' ', 'g'), '')::VARCHAR AS nem_title
FROM stg_bls_crosswalk
ORDER BY onet_soc_code, nem_code;

CREATE OR REPLACE TABLE occupation_market AS
SELECT
    x.onet_soc_code,
    x.onet_soc_title,
    m.*
FROM onet_bls_crosswalk AS x
JOIN bls_market AS m USING (nem_code)
ORDER BY x.onet_soc_code, m.nem_code;
