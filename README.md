## WELCOME TO WOMBO.AI'S PAINT BACK-END MICROSERVICE


**Please run `bash ./setup_pre_commit.sh` at the root of this repository to set up pre-commit hooks compatible with the linting and formatting done on PRs via Github Actions.**

This back-end is served with fastapi and forms the point-of-contact for client apps to interact with WOMBO services. At a high-level, we have tests/ (auto-run Pytests), test_output/ (helper files for tests), and wombo/ (main production code). wombo/ at a high level is in-turn split into api/ (end-points for the clients to interact with), caching_layer/ (a custom in-memory and redis caching layer used across the repository as a Python decorator), general_helpers/ and utils/ (helper functions), graphql_entities/ (resolvers that power the core logic of our graphQL interface), services/ (main "business" logic to interact with tasks, NFTs etc.), and resources/ + scripts/.
