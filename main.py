import pandas as pd
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from openai import OpenAI
import re
import ast


client = OpenAI(api_key="KEY")


df = pd.read_csv("dataset.csv")

def create_distance_matrix(coords):
    matrix = []
    for i in range(len(coords)):
        row = []
        for j in range(len(coords)):
            dist = math.sqrt(
                (coords[i][0] - coords[j][0])**2 +
                (coords[i][1] - coords[j][1])**2
            )
            row.append(int(dist))
        matrix.append(row)
    return matrix


def solve_tsp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[
            manager.IndexToNode(from_index)
        ][
            manager.IndexToNode(to_index)
        ]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    route = []
    index = routing.Start(0)

    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    route.append(manager.IndexToNode(index))
    return route


def get_llm_route(coords):
    prompt = f"""
Solve Traveling Salesman Problem.

Cities:
{coords}

Return ONLY a Python list like:
[0,1,2,...,0]

No explanation.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def parse_llm_output(output):
    match = re.search(r"\[(.*?)\]", output)
    if match:
        try:
            return ast.literal_eval("[" + match.group(1) + "]")
        except:
            return []
    return []


def calculate_distance(route, matrix):
    if not route or len(route) < 2:
        return None
    dist = 0
    for i in range(len(route)-1):
        dist += matrix[route[i]][route[i+1]]
    return dist


NUM_CITIES = 12   
NUM_INSTANCES = 5

results = []

for i in range(NUM_INSTANCES):
    row = df.iloc[i]

    coords = []
    for j in range(1, NUM_CITIES + 1):
        x = row.get(f'City_{j}_X')
        y = row.get(f'City_{j}_Y')
        if pd.notna(x) and pd.notna(y):
            coords.append((float(x), float(y)))

    matrix = create_distance_matrix(coords)

    # OR-Tools
    or_route = solve_tsp(matrix)
    or_distance = calculate_distance(or_route, matrix)

    # LLM
    llm_output = get_llm_route(coords)
    print(f"\nInstance {i} LLM Output:", llm_output)

    llm_route = parse_llm_output(llm_output)
    llm_distance = calculate_distance(llm_route, matrix)

    print("OR Distance:", or_distance)
    print("LLM Distance:", llm_distance)

    results.append({
        "Instance": i,
        "Cities": len(coords),
        "OR Distance": or_distance,
        "LLM Distance": llm_distance
    })


results_df = pd.DataFrame(results)
print("\nFINAL RESULTS:")
print(results_df)

results_df.to_csv("results.csv", index=False)