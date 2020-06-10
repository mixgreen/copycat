# Git workflow

This project uses the [GitHub Flow](https://guides.github.com/introduction/flow/) branching workflow.
Hence, the rolling release branch is `master`.
 
## Branches

We recommend the following naming conventions for branches:

- `feature/*` for developing new features.
- `fix/*` for bug fixes.

## Testing

DAX is shipped with unit tests and default configurations for mypy type checking and flake8 style checking.
The unit tests can run without additional libraries.
Mypy and flake8 need to be installed separately.
Tests can be executed by using the following commands:

```shell
python3 -m unittest
mypy dax/ test/
flake8
```

## Merging

When issuing a merge request into master, we recommend to rebase first or to use the `--no-ff` flag when merging.

## Releases

The `master` branch will always contain the latest production code (rolling release).
Releases are tags and have a two or three-component version number: major, minor, and optionally micro.
See the following examples.

- `v3.0` for a major release with breaking changes.
- `v3.1` for a minor release.
- `v3.1.1` for a micro release containing bug fixes.
