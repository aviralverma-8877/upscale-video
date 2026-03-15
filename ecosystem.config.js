module.exports = {
  apps: [
    {
      name: "upscale-video",
      script: "C:\\Users\\avira\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe",
      args: "app.py",
      cwd: "C:\\Users\\avira\\Documents\\upscale-video",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
