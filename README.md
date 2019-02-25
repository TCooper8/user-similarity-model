# Similar Users Challenge

## Setup

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