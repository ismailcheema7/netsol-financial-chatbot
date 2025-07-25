import gradio as gr
from functions import create_interface
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from gradio.routes import App as GradioApp
from pydantic import BaseModel
from rag_pipeline import rag_agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Annotated
import os
import secrets
from pymongo import MongoClient
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

security = HTTPBasic()

load_dotenv()
api_key = os.getenv("API_KEY")
mongo_uri = os.getenv("MONGODB_URI")

def check(api: str = Header(...)):
    if api != api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
def get_secret_key():
    # Check environment variable first
    secret = os.getenv("SECRET_KEY")
    if secret:
        return secret
    
SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
client = MongoClient(mongo_uri)
db = client["Netsol-Chatbot"]
users_collection = db["users"]
tokens_collection = db["tokens"]

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

hashed_pw = pwd_context.hash("ismail")
users_collection.insert_one({
    "username": "ismail",
    "hashed_password": hashed_pw,
    "disabled": False
})

# --- Pydantic Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    disabled: bool | None = None

class UserInDB(User):
    username:str
    hashed_password: str

class ChatRequest(BaseModel):
    user_input: str
    user_id: str = None

class ChatResponse(BaseModel):
    answer: str

class LoginForm(BaseModel):
    username: str
    password: str


# --- Auth Utilities ---
def verify_password(plain_password, hashed_password):
    print(f"Comparing {plain_password} with {hashed_password}")  # For debugging
    return pwd_context.verify(plain_password, hashed_password) 

def get_password_hash(password):
    """Hash a password using the configured hashing scheme."""
    user_pass = users_collection.find_one({"hashed_password": password})
    return user_pass["hashed_password"] if user_pass else None

def get_user(username: str):
    print(f"Attempting to find user: {username}")
    user_data = users_collection.find_one({"username": username})
    if user_data:
        print(f"Found user: {user_data}")
        return UserInDB(**user_data)
    
    print(f"User {username} not found in {users_collection}")
    return None


def authenticate_user(username: str, password: str):
    print(f"Authenticating user: {username}")
    user = get_user(username)
    if not user:
        print(f"User {username} not found in the database.")
        return False
    
    if not verify_password(password, user.hashed_password):  # Ensure this compares the passwords properly
        print(f"Password mismatch for user: {username}")
        return False

    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    # Save token in MongoDB
    try:
        tokens_collection.insert_one({
            "token": encoded_jwt,
            "username": data.get("sub"),
            "created_at": datetime.now(timezone.utc),
            "expires_at": expire
        })
    except Exception as e:
        print(f"Error inserting token into MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="Token saving failed")

    return encoded_jwt

# --- Current User Dependency ---
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            print("no username wehn getting user")
            raise credentials_exception

        # Check token exists in MongoDB (revoked tokens are not found)
        if not tokens_collection.find_one({"token": token}):
            print("Token not found when geting user")
            raise HTTPException(status_code=401, detail="Token revoked or not found")

        token_data = TokenData(username=username)

    except JWTError:
        print("JWT error when getting user")
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        print("no user when getting user")
        raise credentials_exception
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    if current_user.disabled:
        print("User disabled when getting active user")
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = authenticate_user(form_data.username, form_data.password)
    except Exception as e:
        print(f"Error during authentication: {e} when logging for toke")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    if not user:
        print("user not found")
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.disabled:
        print(f"User {user.username} is disabled.")
        raise HTTPException(
            status_code=400,
            detail="Inactive user",
        )
    
    print("✅ Fetched user:", user.username)  # <-- now it's safe

    try:
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    except Exception as e:
        print(f"Error during token generation: {e}")
        raise HTTPException(status_code=500, detail="Token generation failed")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/logout")
async def logout(token: Annotated[str, Depends(oauth2_scheme)]):
    deleted = tokens_collection.delete_one({"token": token})
    if deleted.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Token not found or already revoked")
    return {"message": "Token successfully revoked. You have been logged out."}


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user

@app.get("/users/me/items")
async def read_own_items(current_user: Annotated[User, Depends(get_current_active_user)]):
    return [{"item_id": "Foo", "owner": current_user.username}]

@app.get("/tokens")
async def list_tokens():
    """ Debug endpoint to list saved tokens """
    tokens = list(tokens_collection.find({}, {"_id": 0}))
    return {"tokens": tokens}


@app.get("/")
def root():
    return {"message": "FastAPI is running, go to /docs or /chat"}

demo = create_interface()
app = gr.mount_gradio_app(app, demo, path="/gradio")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Endpoint
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(oauth2_scheme)])
def chat(request: ChatRequest):
    try:
        messages = [HumanMessage(content=request.user_input)]
        result = rag_agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        return ChatResponse(answer=answer)

    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    