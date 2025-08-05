```mermaid
flowchart LR
 subgraph s1["Trading System"]
        n5["Financial Data Port"]
        n6["Broker Port"]
        n7["New Data Port"]
        n8["Trading Engine"]
  end
    n6 --> n2["Broker"]
    n3["Financial Data"] --> n5
    n4["News Outlet"] --> n7
    n5 -- Bar Data --> n8
    n7 -- News Data --> n8
    n8 -- Orders --> n6
