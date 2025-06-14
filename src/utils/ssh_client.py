"""
SSH client utilities for remote file operations.
"""
import paramiko
from scp import SCPClient
import logging
import os

logger = logging.getLogger(__name__)

class SSHClient:
    def __init__(self, hostname: str, username: str, password: str, port: int = 22):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port
        self.client = None
        self.transport = None

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.hostname,
            username=self.username,
            password=self.password,
            port=self.port
        )
        self.transport = self.client.get_transport()
        if self.transport is None:
            raise RuntimeError("Failed to get SSH transport for SCP connection.")

    def close(self):
        if self.transport:
            self.transport.close()
        if self.client:
            self.client.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_file(self, remote_path: str, local_path: str):
        """Get a file from remote server using SCP"""
        if self.transport is None:
            raise RuntimeError("SSH transport is not initialized")
        try:
            with SCPClient(self.transport) as scp:
                scp.get(remote_path, local_path)
        except Exception as e:
            logger.error(f"Error in SCP file transfer: {str(e)}")
            raise 