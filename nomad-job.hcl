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
        cpu    = 500 
        memory = 256 
      }

      env {
        FLASK_APP = "HISKIN_FA"  
      }

    }
  }
}
