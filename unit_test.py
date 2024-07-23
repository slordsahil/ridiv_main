import pytest
from fastapi.testclient import TestClient
from ridiv_assignment import my_app,file_chunks

client=TestClient(my_app)

def cache_clear():
    file_chunks.clear()
    
    yield
    
def testing_upload_file():
    with open("test_file.txt", "rb") as f:
        response = client.post("/upload_file/", files={"file": f})
        
        with open("test_file.txt", "rb") as f:
            response = client.post("/upload_file/", files={"file": f})

        assert response.status_code == 200
        assert response.json() == ["successfully uploaded"]    
        
    
def testing_ask_me():
    request_data = {"question": "What is the test?"}
    response = client.post("/ask_me/", json=request_data)

    assert response.status_code == 200
    assert "response" in response.json()
    assert "similarity" in response.json()
    
    
def testing_delete_data():
    
    response = client.post("/delete_my_pdf_data/")
    
    assert response.status_code == 200
    assert response.json() == {"status": "All data deleted successfully"}  
