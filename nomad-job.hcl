job "flask-app" {
  datacenters = ["dc1"]

  group "web-group" {
    count = 1

    task "flask-task" {
      driver = "exec" 

      config {
        command = "/usr/bin/python3"
        args    = ["-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
      }

      resources {
        cpu    = 100   # 0.1 vCPU
        memory = 256   # 256 MB RAM
        disk   = 2000  # 2 GB disk
      }

      env {
        FLASK_APP = "HISKIN_FA"  
      }

    }
  }
}