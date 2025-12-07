# JupyterHub server proxy configuration for MyBinder
# This tells JupyterHub to allow proxying to port 8000

c.ServerProxy.servers = {
    'hooplytics': {
        'command': ['uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'],
        'port': 8000,
        'timeout': 30,
        'absolute_url': False,
        'launcher_entry': {
            'enabled': True,
            'title': 'Hooplytics',
            'icon_path': '/home/jovyan/frontend/public/vite.svg'
        }
    }
}
