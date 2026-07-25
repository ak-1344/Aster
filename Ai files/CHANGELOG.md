# Changelog

All notable changes to ASTER will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-25

### Added
- Execution graph data structures (backend/app/execution_graph/execution_graph.py) representing Planner output as ordered nodes with explicit dependencies.
- Scheduler execution (backend/app/scheduler/scheduler.py) that consumes execution graphs and runs nodes in dependency order via Phase 3 entrypoints.
- Response composer (backend/app/response_composer/response_composer.py) that merges node outputs into structured response objects.
- Execution-engine smoke tests (backend/tests/test_execution_engine_smoke.py) validating end-to-end workflow execution for segmentation and descriptive queries.
- Execution graph topological sort for dependency-ordered node execution.
- Model registry (backend/app/model_registry) for segmentation algorithm discovery and selection.

### Changed
- Scheduler now respects node dependencies and never reorders dependent nodes per architecture.md Section 29 rules.
- Response composer now combines statistics, recommendations, and visual outputs into single structured responses.
- Execution graph infers dependencies from planner step inputs/outputs using substring matching.
- Planner now includes classify_intent function to route queries as full_workflow, explanation_only, or eda_only.

### Fixed
- Scheduler now passes context by reference so feature_engineering output (customer_features) is visible to downstream segmentation/evaluation/recommendation/visualization nodes.

### Implementation Details
- ExecutionGraph uses Kahn's algorithm for topological sorting with fallback to original order on cycle detection.
- Smoke tests validate planner → execution_graph → scheduler → response_composer pipeline for both segmentation and descriptive workflows.
- All execution-engine tests pass with the CC GENERAL dataset.

## [0.4.0] - 2026-07-25

### Added
- Rule-based planner entrypoint in backend/app/planner/planner.py for analytical workflow selection.
- Query normalization and intent routing in backend/app/query_manager/query_manager.py.
- Context-building support for planner inputs in backend/app/context_builder/context_builder.py.
- Planner smoke tests for segmentation and descriptive queries.

### Changed
- Query manager now normalizes user queries and routes them through the planner.
- Context builder extracts structured context (intent, filters, entities) for planner consumption.
- Planner returns executable workflows targeting Phase 3 analytical nodes.

### Implementation Details
- Planner uses rule-based logic to select appropriate analytical nodes based on query intent.
- Context builder provides structured input for planner decision-making.
- Smoke tests validate planner output for segmentation and descriptive/EDA-only workflows.

## [0.3.0] - 2026-07-25

### Added
- Phase 3 analytical node implementations for analytics, EDA, segmentation, recommendation, evaluation, and visualization.
- Backend smoke tests covering each analytical node using the CC GENERAL dataset.

### Changed
- Moved the implementation tracker from Phase 3 to Phase 4 planning and updated the current project status accordingly.

## [0.2.0] - 2026-07-25

### Added
- Reusable analytical node entrypoints in backend/app/nodes for descriptive statistics, exploratory summaries, customer segmentation, rule-based recommendations, evaluation metrics, and scatter-style visualization payloads.
- Backend smoke tests to verify each node independently against the current dataset.

### Changed
- Marked Phase 3 as completed in the task tracker and aligned the roadmap with the implemented repository state.

## [0.1.0] - 2026-07-25

### Added
- Phase 0 backend/app scaffold, dependency manifest, environment example, and frontend placeholder.
- Dataset-understanding utility and EDA node wrapper for the selected customer-level dataset.
- Dataset-understanding documentation covering schema, identifiers, missing values, and datatypes.

### Changed
- Promoted Phase 0 and Phase 1 checklists to completed milestones in the task tracker.
- Synchronized the project status documentation with the current repository state.
