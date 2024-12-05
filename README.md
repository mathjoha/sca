<a href="https://creativecommons.org/licenses/by-nc/4.0/"><img decoding="async" loading="eager" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" width="71" height="25" align="right"></a>

# Template

This is a template README for setting up new repositories within the
*Modern36* organization on GitHub.

It contains a basic structure that is nice to have for new projects and
it has two workflows set up in the `.github/workflows` directory:

 1. Auto assign:
    - Automatically add assignees all opened issues and pulls requests
    that do not already have an assignee.
    - By default this assignee is @mathjoha
 2. Link to project:
    - Automatically add all opened and reopened issues to the project view.
    - Ignores issues labelled as `bug`


## Usage

1. m36_utils requirement
   - Remove it if not needed
   - Make sure that it is accessible if it is needed
2. Install pre-commit functionalities:
   `pre-commit install`
3. Update this file to describe the repository
4. Commit


# License

The code is published under a Creative Commons Attribution-NonCommercial
4.0 International license [CC BY-NC 4.0 license](/LICENSE).
