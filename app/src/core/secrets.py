"""
Veilige opslag van API credentials en geheime sleutels
"""
import os
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger
import yaml
from cryptography.fernet import Fernet
import base64


class SecretsManager:
    """Beheer van API credentials en geheime sleutels"""
    
    def __init__(self, secrets_file: str = "secrets.yaml", encrypted_file: str = "secrets.enc"):
        self.secrets_file = Path(secrets_file)
        self.encrypted_file = Path(encrypted_file)
        self.key_file = Path("encryption.key")
        self._secrets = {}
        self._load_secrets()
    
    def _generate_key(self) -> bytes:
        """Genereer encryption key"""
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        logger.info("Nieuwe encryption key gegenereerd")
        return key
    
    def _load_key(self) -> bytes:
        """Laad encryption key van environment variable of genereer nieuwe"""
        # Probeer eerst environment variable (production)
        env_key = os.getenv('TRADING_BOT_ENCRYPTION_KEY')
        if env_key:
            try:
                # Validate key format
                if len(env_key) != 44:  # Fernet key length
                    raise ValueError("Invalid key length")
                return env_key.encode()
            except Exception as e:
                logger.error(f"Invalid encryption key in environment: {e}")
                raise ValueError("Invalid encryption key format")
        
        # Check if we're in production
        if os.getenv('TRADING_BOT_ENV') == 'production':
            logger.error("Production environment requires TRADING_BOT_ENCRYPTION_KEY environment variable")
            raise ValueError("Encryption key not found in production environment")
        
        # Fallback naar key file (alleen voor development)
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                return f.read()
        
        # Genereer nieuwe key (alleen voor development)
        logger.warning("Generating new encryption key - this should only happen in development!")
        return self._generate_key()
    
    def _encrypt_secrets(self, secrets: Dict[str, Any]) -> None:
        """Encrypt secrets en sla op"""
        try:
            key = self._load_key()
            fernet = Fernet(key)
            
            # Converteer naar string
            secrets_str = yaml.dump(secrets)
            secrets_bytes = secrets_str.encode()
            
            # Encrypt
            encrypted_data = fernet.encrypt(secrets_bytes)
            
            # Sla op
            with open(self.encrypted_file, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info("Secrets geëncrypteerd en opgeslagen")
            
        except Exception as e:
            logger.error(f"Fout bij encrypten secrets: {e}")
    
    def _decrypt_secrets(self) -> Dict[str, Any]:
        """Decrypt secrets"""
        try:
            if not self.encrypted_file.exists():
                return {}
            
            key = self._load_key()
            fernet = Fernet(key)
            
            # Lees encrypted data
            with open(self.encrypted_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted_data = fernet.decrypt(encrypted_data)
            secrets_str = decrypted_data.decode()
            
            # Parse YAML
            secrets = yaml.safe_load(secrets_str)
            return secrets or {}
            
        except Exception as e:
            logger.error(f"Fout bij decrypten secrets: {e}")
            return {}
    
    def _load_secrets(self) -> None:
        """Laad secrets van bestand of environment variables"""
        # Probeer eerst encrypted file
        self._secrets = self._decrypt_secrets()
        
        # Als dat niet werkt, probeer plain text file
        if not self._secrets and self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    self._secrets = yaml.safe_load(f) or {}
                logger.warning("Secrets geladen van plain text file - overweeg encryptie")
            except Exception as e:
                logger.error(f"Fout bij laden secrets file: {e}")
        
        # Fallback naar environment variables
        if not self._secrets:
            self._secrets = self._load_from_env()
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Laad secrets van environment variables"""
        secrets = {}
        
        # Kraken API
        if os.getenv('KRAKEN_API_KEY'):
            secrets['kraken'] = {
                'api_key': os.getenv('KRAKEN_API_KEY'),
                'api_secret': os.getenv('KRAKEN_API_SECRET', ''),
                'sandbox': os.getenv('KRAKEN_SANDBOX', 'true').lower() == 'true'
            }
        
        # Telegram
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            secrets['telegram'] = {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
            }
        
        # Database
        if os.getenv('DATABASE_URL'):
            secrets['database'] = {
                'url': os.getenv('DATABASE_URL')
            }
        
        if secrets:
            logger.info("Secrets geladen van environment variables")
        
        return secrets
    
    def get_secret(self, service: str, key: str, default: Any = None) -> Any:
        """
        Krijg specifieke secret (nooit loggen)
        
        Audit trail:
        - Logs secret access (key-id only, never value)
        - Fails closed: returns None if secret not found in production
        - No secret values are ever logged or exposed
        """
        try:
            secret = self._secrets.get(service, {}).get(key, default)
            
            # Audit trail: log access but never the value
            if secret:
                logger.debug(f"Secret '{service}.{key}' accessed (value hidden)")
            else:
                logger.warning(f"Secret '{service}.{key}' not found")
            
            # Fail-closed behavior in production
            if not secret and os.getenv('TRADING_BOT_ENV') == 'production':
                logger.error(f"CRITICAL: Secret '{service}.{key}' not found in production - failing closed")
                return None
            
            return secret
        except Exception as e:
            logger.error(f"Fout bij ophalen secret {service}.{key}: {e}")
            # Fail-closed: return None in production on error
            if os.getenv('TRADING_BOT_ENV') == 'production':
                return None
            return default
    
    def set_secret(self, service: str, key: str, value: Any) -> None:
        """Zet secret waarde"""
        if service not in self._secrets:
            self._secrets[service] = {}
        
        self._secrets[service][key] = value
        logger.debug(f"Secret {service}.{key} bijgewerkt")
    
    def save_secrets(self, encrypt: bool = True) -> None:
        """Sla secrets op"""
        if encrypt:
            self._encrypt_secrets(self._secrets)
        else:
            # Plain text (alleen voor development)
            with open(self.secrets_file, 'w') as f:
                yaml.dump(self._secrets, f, default_flow_style=False)
            logger.warning("Secrets opgeslagen als plain text")
    
    def get_kraken_credentials(self) -> Dict[str, Any]:
        """Krijg Kraken API credentials"""
        return {
            'api_key': self.get_secret('kraken', 'api_key', ''),
            'api_secret': self.get_secret('kraken', 'api_secret', ''),
            'sandbox': self.get_secret('kraken', 'sandbox', True)
        }
    
    def get_telegram_credentials(self) -> Dict[str, Any]:
        """Krijg Telegram credentials"""
        return {
            'bot_token': self.get_secret('telegram', 'bot_token', ''),
            'chat_id': self.get_secret('telegram', 'chat_id', '')
        }
    
    def setup_interactive(self) -> None:
        """Interactieve setup van secrets"""
        print("🔐 Secrets Manager Setup")
        print("=" * 50)
        
        # Kraken API
        print("\n📊 Kraken API Setup:")
        api_key = input("Kraken API Key (optioneel): ").strip()
        if api_key:
            api_secret = input("Kraken API Secret: ").strip()
            sandbox = input("Sandbox mode (y/n) [y]: ").strip().lower() != 'n'
            
            self.set_secret('kraken', 'api_key', api_key)
            self.set_secret('kraken', 'api_secret', api_secret)
            self.set_secret('kraken', 'sandbox', sandbox)
        
        # Telegram
        print("\n📱 Telegram Setup:")
        bot_token = input("Telegram Bot Token (optioneel): ").strip()
        if bot_token:
            chat_id = input("Telegram Chat ID: ").strip()
            
            self.set_secret('telegram', 'bot_token', bot_token)
            self.set_secret('telegram', 'chat_id', chat_id)
        
        # Sla op
        encrypt = input("\n🔒 Encrypt secrets (y/n) [y]: ").strip().lower() != 'n'
        self.save_secrets(encrypt=encrypt)
        
        print("✅ Secrets setup voltooid!")


# Global instance
secrets_manager = SecretsManager()


def get_kraken_credentials() -> Dict[str, Any]:
    """Convenience functie voor Kraken credentials"""
    return secrets_manager.get_kraken_credentials()


def get_telegram_credentials() -> Dict[str, Any]:
    """Convenience functie voor Telegram credentials"""
    return secrets_manager.get_telegram_credentials()


if __name__ == "__main__":
    # Interactive setup
    secrets_manager.setup_interactive()
