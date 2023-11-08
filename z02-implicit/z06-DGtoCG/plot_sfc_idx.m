sfc_data = readmatrix('z23-nozzle/nozzle_scale1_d4_D4_r1.msh_sfc_data_out.txt');
sorted_data = sortrows(sfc_data, 4);
idx_to_plot = 1:size(sorted_data, 1);
figure();
scatter3(sorted_data(idx_to_plot,1), sorted_data(idx_to_plot,3), sorted_data(idx_to_plot,2),ones(size(sorted_data(1)))*30, sorted_data(idx_to_plot,4), 'filled')
colorbar
axis equal
hold on
plot3(sorted_data(idx_to_plot,1), sorted_data(idx_to_plot,3), sorted_data(idx_to_plot,2), 'Color', [.7,.7,.7])
axis off