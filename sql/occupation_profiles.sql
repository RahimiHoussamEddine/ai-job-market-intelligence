-- O*NET 30.3 identity, alias, task, profile, and evaluation-qrel tables.
-- Execute after staging normalization and after sql/skill_vectors.sql.
-- Related Occupations is deliberately materialized separately and is never
-- referenced by occupation_profiles or any other searchable feature table.

CREATE OR REPLACE TABLE occupations AS
SELECT
    o.onet_soc_code::VARCHAR AS onet_soc_code,
    left(o.onet_soc_code, 7)::VARCHAR AS base_soc_code,
    left(o.onet_soc_code, 2)::VARCHAR AS soc_major_group,
    trim(o.title)::VARCHAR AS canonical_title,
    regexp_replace(trim(o.description), '\s+', ' ', 'g')::VARCHAR AS description,
    try_cast(z.job_zone AS INTEGER) AS job_zone,
    coalesce(
        try_cast(z.date AS DATE),
        try_strptime(z.date, '%m/%Y')::DATE,
        try_strptime(z.date, '%Y-%m')::DATE
    ) AS job_zone_date,
    nullif(regexp_replace(trim(z.domain_source), '\s+', ' ', 'g'), '')::VARCHAR
        AS job_zone_domain_source
FROM stg_occupation_data AS o
LEFT JOIN stg_job_zones AS z USING (onet_soc_code)
ORDER BY o.onet_soc_code;

CREATE OR REPLACE TABLE aliases AS
SELECT
    sha256(
        j.onet_soc_code || chr(31) ||
        lower(regexp_replace(trim(j.job_title), '\s+', ' ', 'g')) || chr(31) ||
        coalesce(regexp_replace(trim(j.source), '\s+', ' ', 'g'), '')
    )::VARCHAR AS alias_id,
    j.onet_soc_code::VARCHAR AS onet_soc_code,
    regexp_replace(trim(j.job_title), '\s+', ' ', 'g')::VARCHAR AS job_title,
    lower(regexp_replace(trim(j.job_title), '\s+', ' ', 'g'))::VARCHAR AS normalized_job_title,
    nullif(regexp_replace(trim(j.short_title), '\s+', ' ', 'g'), '')::VARCHAR AS short_title,
    nullif(regexp_replace(trim(j.source), '\s+', ' ', 'g'), '')::VARCHAR AS sources
FROM stg_job_titles AS j
ORDER BY onet_soc_code, normalized_job_title, job_title, sources;

CREATE OR REPLACE TABLE tasks AS
SELECT
    t.onet_soc_code::VARCHAR AS onet_soc_code,
    trim(t.task_id)::VARCHAR AS task_id,
    regexp_replace(trim(t.task), '\s+', ' ', 'g')::VARCHAR AS task,
    lower(regexp_replace(trim(t.task), '\s+', ' ', 'g'))::VARCHAR AS normalized_task,
    nullif(trim(t.task_type), '')::VARCHAR AS task_type,
    try_cast(replace(t.incumbents_responding, ',', '') AS BIGINT) AS incumbents_responding,
    coalesce(
        try_cast(t.date AS DATE),
        try_strptime(t.date, '%m/%Y')::DATE,
        try_strptime(t.date, '%Y-%m')::DATE
    ) AS task_date,
    nullif(regexp_replace(trim(t.domain_source), '\s+', ' ', 'g'), '')::VARCHAR AS domain_source
FROM stg_task_statements AS t
ORDER BY onet_soc_code, task_id;

CREATE OR REPLACE TABLE occupation_profiles AS
WITH deduplicated_aliases AS (
    SELECT onet_soc_code, normalized_job_title, min(job_title) AS job_title
    FROM aliases
    GROUP BY onet_soc_code, normalized_job_title
),
alias_rollup AS (
    SELECT
        o.onet_soc_code,
        string_agg(
            a.job_title,
            chr(10) ORDER BY a.normalized_job_title, a.job_title
        ) FILTER (
            WHERE a.normalized_job_title <> lower(o.canonical_title)
        ) AS alias_text,
        count(*) FILTER (
            WHERE a.normalized_job_title <> lower(o.canonical_title)
        )::BIGINT AS alias_count
    FROM occupations AS o
    LEFT JOIN deduplicated_aliases AS a USING (onet_soc_code)
    GROUP BY o.onet_soc_code
),
deduplicated_core_tasks AS (
    SELECT onet_soc_code, normalized_task, min(task_id) AS task_id, min(task) AS task
    FROM tasks
    WHERE task_type = 'Core'
    GROUP BY onet_soc_code, normalized_task
),
task_rollup AS (
    SELECT
        onet_soc_code,
        string_agg(task, chr(10) ORDER BY task_id, normalized_task)::VARCHAR AS core_task_text,
        count(*)::BIGINT AS core_task_count
    FROM deduplicated_core_tasks
    GROUP BY onet_soc_code
),
rating_rollup AS (
    SELECT
        onet_soc_code,
        count(*) FILTER (WHERE observed)::BIGINT AS structured_observed_count
    FROM skill_ratings
    GROUP BY onet_soc_code
)
SELECT
    o.*,
    concat_ws(chr(10), o.canonical_title, a.alias_text)::VARCHAR AS title_text,
    concat_ws(chr(10), o.description, t.core_task_text)::VARCHAR AS work_text,
    coalesce(t.core_task_count, 0)::BIGINT AS core_task_count,
    coalesce(a.alias_count, 0)::BIGINT AS alias_count,
    coalesce(r.structured_observed_count, 0)::BIGINT AS structured_observed_count
FROM occupations AS o
LEFT JOIN alias_rollup AS a USING (onet_soc_code)
LEFT JOIN task_rollup AS t USING (onet_soc_code)
LEFT JOIN rating_rollup AS r USING (onet_soc_code)
ORDER BY o.onet_soc_code;

CREATE OR REPLACE TABLE related_qrels AS
SELECT
    r.onet_soc_code::VARCHAR AS query_onet_soc_code,
    r.related_onet_soc_code::VARCHAR AS candidate_onet_soc_code,
    trim(r.relatedness_tier)::VARCHAR AS relatedness_tier,
    try_cast(r.index AS INTEGER) AS relatedness_index,
    CASE trim(r.relatedness_tier)
        WHEN 'Primary-Short' THEN 3
        WHEN 'Primary-Long' THEN 2
        WHEN 'Supplemental' THEN 1
    END::INTEGER AS relevance_grade
FROM stg_related_occupations AS r
ORDER BY query_onet_soc_code, relatedness_index, candidate_onet_soc_code;
