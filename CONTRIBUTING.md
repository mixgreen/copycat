# Git workflow

This project uses the [GitHub Flow](https://guides.github.com/introduction/flow/) branching workflow.
Hence, the rolling release branch is `master`.
 
## Branches

We recommend the following naming conventions for branches:

- `feature/*` for developing new features.
- `fix/*` for bug fixes.

## Testing

DAX is shipped with unit tests and default configurations for mypy type checking and flake8 style checking.
Unit tests are based on the standard Python unittest library.
To run unit tests, use pytest or Python unittest test discovery.
pytest, mypy, and flake8 need to be installed separately.
Tests can be executed by using the following commands:

```shell
pytest
mypy
flake8
```

## Merging

To merge into master, please issue a merge request.

## Releases

The `master` branch will always contain the latest production code (rolling release).
Releases are tags and have a two or three-component version number: major, minor, and optionally micro.
See the following examples.

- `v3.0` for a major release with breaking changes.
- `v3.1` for a minor release.
- `v3.1.1` for a micro release containing bug fixes.
