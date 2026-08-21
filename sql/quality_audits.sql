-- Strict, machine-readable release gates.  A valid snapshot returns no rows
-- from quality_audit_failures.  O*NET counts are release-30.3 pins, not generic
-- assumptions and must change only with a new source/experiment version.

CREATE OR REPLACE VIEW quality_audit_results AS
SELECT 'release.occupation_data_rows' AS check_name,
       abs((SELECT count(*) FROM stg_occupation_data) - 1016)::BIGINT AS failure_count,
       'O*NET 30.3 requires 1,016 occupations' AS detail
UNION ALL
SELECT 'release.job_titles_rows', abs((SELECT count(*) FROM stg_job_titles) - 57543),
       'O*NET 30.3 requires 57,543 job-title rows'
UNION ALL
SELECT 'release.task_statements_rows', abs((SELECT count(*) FROM stg_task_statements) - 18796),
       'O*NET 30.3 requires 18,796 task rows'
UNION ALL
SELECT 'release.transferable_skills_rows', abs((SELECT count(*) FROM stg_transferable_skills) - 44700),
       'O*NET 30.3 requires 44,700 transferable-skill rows'
UNION ALL
SELECT 'release.knowledge_rows', abs((SELECT count(*) FROM stg_knowledge) - 59004),
       'O*NET 30.3 requires 59,004 knowledge rows'
UNION ALL
SELECT 'release.job_zones_rows', abs((SELECT count(*) FROM stg_job_zones) - 923),
       'O*NET 30.3 requires 923 four-level Job Zone rows'
UNION ALL
SELECT 'release.related_occupations_rows', abs((SELECT count(*) FROM stg_related_occupations) - 18460),
       'O*NET 30.3 requires 18,460 related-occupation rows'
UNION ALL
SELECT 'occupations.duplicate_key', count(*) - count(DISTINCT onet_soc_code),
       'Occupation code must be unique' FROM occupations
UNION ALL
SELECT 'occupations.job_zone_range', count(*), 'Job Zone must be null or in current range 1-4'
FROM occupations WHERE job_zone NOT BETWEEN 1 AND 4
UNION ALL
SELECT 'aliases.duplicate_key', count(*) - count(DISTINCT alias_id),
       'Deterministic alias ID must be unique' FROM aliases
UNION ALL
SELECT 'tasks.duplicate_key', count(*) - count(DISTINCT onet_soc_code || chr(31) || task_id),
       'Task business key must be unique' FROM tasks
UNION ALL
SELECT 'skill_ratings.duplicate_key', count(*) - count(DISTINCT
       onet_soc_code || chr(31) || feature_family || chr(31) || element_id || chr(31) || scale_id),
       'Structured rating feature key must be unique' FROM skill_ratings
UNION ALL
SELECT 'skill_ratings.mask', count(*),
       'Unobserved values are null; observed values are within normalized bounds'
FROM skill_ratings
WHERE (observed AND (normalized_value IS NULL OR normalized_value NOT BETWEEN 0 AND 1))
   OR (NOT observed AND normalized_value IS NOT NULL)
   OR (observed AND (recommend_suppress OR not_relevant))
UNION ALL
SELECT 'skill_ratings.precision', count(*),
       'Sample/precision fields must be nonnegative and confidence bounds ordered'
FROM skill_ratings
WHERE sample_size < 0 OR standard_error < 0
   OR ((lower_ci_bound IS NULL) <> (upper_ci_bound IS NULL))
   OR lower_ci_bound > upper_ci_bound
   OR (raw_value IS NOT NULL AND lower_ci_bound IS NOT NULL
       AND raw_value NOT BETWEEN lower_ci_bound AND upper_ci_bound)
UNION ALL
SELECT 'related_qrels.self_match', count(*), 'Evaluation qrels must exclude self matches'
FROM related_qrels WHERE query_onet_soc_code = candidate_onet_soc_code
UNION ALL
SELECT 'related_qrels.duplicate_pair', count(*) - count(DISTINCT
       query_onet_soc_code || chr(31) || candidate_onet_soc_code),
       'Each query/candidate qrel pair must be unique' FROM related_qrels
UNION ALL
SELECT 'related_qrels.grade', count(*), 'Relevance grade must use preregistered 3/2/1 mapping'
FROM related_qrels WHERE relevance_grade NOT IN (1, 2, 3)
UNION ALL
SELECT 'related_qrels.tier_grade', count(*), 'Tier and grade must use the locked 3/2/1 mapping'
FROM related_qrels
WHERE relevance_grade <> CASE relatedness_tier
    WHEN 'Primary-Short' THEN 3 WHEN 'Primary-Long' THEN 2 WHEN 'Supplemental' THEN 1 END
UNION ALL
SELECT 'bls_market.summary_rows', count(*), 'Only detailed BLS Line item rows are allowed'
FROM bls_market WHERE lower(occupation_type) <> 'line item'
UNION ALL
SELECT 'bls_market.duplicate_nem', count(*) - count(DISTINCT nem_code),
       'BLS market row must be unique by NEM code' FROM bls_market
UNION ALL
SELECT 'crosswalk.duplicate_pair', count(*) - count(DISTINCT
       onet_soc_code || chr(31) || nem_code),
       'Official O*NET/NEM code pair must be unique' FROM onet_bls_crosswalk
UNION ALL
SELECT 'occupation_market.cardinality', abs(
       (SELECT count(*) FROM occupation_market) - (SELECT count(*) FROM onet_bls_crosswalk)),
       'Market join must preserve official crosswalk cardinality';

CREATE OR REPLACE VIEW quality_audit_failures AS
SELECT *
FROM quality_audit_results
WHERE failure_count <> 0
ORDER BY check_name;
