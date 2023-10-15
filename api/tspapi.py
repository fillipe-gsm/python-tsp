from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from api.datamodel import Cities, City
from api.tsp import compute_optimal_tour

app = FastAPI()


@app.get("/")
def root():
    return {"Welcome to the TSP API"}


@app.post("/tsp/")
async def tsp(req: Cities):
    city_list: List[City] = req.cities
    (optimal_tour, distance) = compute_optimal_tour(city_list)
    res: dict = {"tour": optimal_tour, "distance": distance}
    return JSONResponse(content=res)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
