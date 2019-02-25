# Similar Users Challenge

## Setup

Datasets:
  - Currently, you will need to add specific datasets to the project.

```
/user-similarity-model
  /datasets
    course_tags.csv
    user_assessment_scores.csv
    user_course_views.csv
    user_interests.csv
  /scripts ...
  /services ...
  docker-compose.yaml
```

The structure above is important for the datasets because it is declared as a volume mount in the docker-compose.

Requirements:
  - docker

### Unix/Linux

```sh
./scripts/bash/run.sh
```

### Windows

```powershell
./scripts/win/run.ps1
```

## REST Api

| resource | description |
|:---------|:------------|
| `/health`| Basic health check to see if the service is running |
| `/users/{user_handle}/similar` | Returns a list of user handles similar to the given user by `user_handle` |