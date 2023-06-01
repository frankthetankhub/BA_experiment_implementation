import data_wrangling as dw

def test_dataframe():
    df_set,  df_lth_bestacc, df_lth_all= dw.load_dataframes()
    print(df_lth_all)
    group = df_lth_all.groupby(["dataset","arch_size", "compression", "patience"])
    missing =0
    for name, data in group:
        if data.shape[0] !=5:
            missing += (5 - data.shape[0])
            print(data.shape)
            print(list(zip(["dataset","arch_size", "compression", "patience"],name)))
            print(data["seed"])
            print("------------------------------")

    print(df_set)
    group = df_set[df_set.workers.eq(0)].groupby(["dataset","arch_size", "zeta_anneal","config"])
    extra =0
    runs = {}
    i=0
    for name, data in group:
        if data.shape[0] >5:
            extra += data.shape[0] -5
            #print(data.shape)
            l =list(zip(["dataset","arch_size", "workers", "zeta_anneal", "config"],name))
            print(name, data["seed"].values)
            # print(l)
            # print(data["seed"])
            # print("------------------------------")
            #l.append()
            runs[i]= l
            i+=1
    print(extra)
    print(df_set)
    df_set = df_set.drop_duplicates(subset=["dataset","arch_size","zeta_anneal","config","workers","seed"])
    print(df_set)
    group = df_set[df_set.workers.eq(0)].groupby(["dataset","arch_size", "zeta_anneal","config", "workers"])
    missing =0
    runs = {}
    i=0
    for name, data in group:
        if data.shape[0] <5:
            missing += abs(5 - data.shape[0])
            #print(data.shape)
            l =list(zip(["dataset","arch_size", "workers", "zeta_anneal", "config"],name))
            print(name, data["seed"].values)
            # print(l)
            # print(data["seed"])
            # print("------------------------------")
            #l.append()
            runs[i]= l
            i+=1
    print(missing)
    
    print(runs)
    print(f"missing runs for set: {missing}")
    group = df_set[df_set.workers.eq(3)].groupby(["dataset","arch_size", "zeta_anneal","config"])
    missing =0
    runs = {}
    i=0
    for name, data in group:
        if data.shape[0] !=5:
            missing += abs(5 - data.shape[0])
            l =list(zip(["dataset","arch_size", "zeta_anneal", "config"],name))
            print(name, data["seed"].values)
            if data["seed"].values.shape[0] < 5:
                l.append(data["seed"].values)
                runs[i]= l
                i+=1
    print(runs)
    print(f"missing runs for wasap: {missing}")

if __name__=="__main__":
    test_dataframe()