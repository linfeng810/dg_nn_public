import main
from torch.profiler import profile, ProfilerActivity, record_function
import torch 

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/main'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    prof.step()
    main.main()
    prof.step()

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))