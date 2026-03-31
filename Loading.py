import pyarrow.parquet as pq
import pyarrow as pa

pf = pq.ParquetFile("F:\QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494_LR.parquet")

for i, batch in enumerate(pf.iter_batches(batch_size=10000)):
    table = pa.Table.from_batches([batch])
    print("Parquet loaded")
    pq.write_table(table, f"run_2_chunk_{i}.parquet")
    print(f"chunk_{i}.parquet saved")