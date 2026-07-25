# Changelog

All notable changes to ASTER will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3 analytical node implementations for analytics, EDA, segmentation, recommendation, evaluation, and visualization.
- Backend smoke tests covering each analytical node using the CC GENERAL dataset.

### Changed
- Moved the implementation tracker from Phase 3 to Phase 4 planning and updated the current project status accordingly.

## [0.3.0] - 2026-07-25

### Added
- Reusable analytical node entrypoints in backend/app/nodes for descriptive statistics, exploratory summaries, customer segmentation, rule-based recommendations, evaluation metrics, and scatter-style visualization payloads.
- Backend smoke tests to verify each node independently against the current dataset.

### Changed
- Marked Phase 3 as completed in the task tracker and aligned the roadmap with the implemented repository state.

## [0.2.0] - 2026-07-25

### Added
- Reusable feature engineering utility at backend/utils/feature_engineering.py.
- Feature engineering node wrapper at backend/app/nodes/feature_engineering_node.py.
- Generated processed dataset artifact backend/data/processed/customer_features.csv.

### Changed
- Promoted Phase 2 to a completed milestone in the task tracker.
- Updated project status and next actions to target Phase 3 analytical nodes.

## [0.1.0] - 2026-07-25

### Added
- Phase 0 backend/app scaffold, dependency manifest, environment example, and frontend placeholder.
- Dataset-understanding utility and EDA node wrapper for the selected customer-level dataset.
- Dataset-understanding documentation covering schema, identifiers, missing values, and datatypes.

### Changed
- Promoted Phase 0 and Phase 1 checklists to completed milestones in the task tracker.
- Synchronized the project status documentation with the current repository state.
