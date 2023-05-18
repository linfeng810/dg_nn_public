function plot_tetra(ax, x, plab, flab, vlab, shrinkage, x_node)
% take in a figure axe handle ax
% and tetrahedron corners coordinate x
% plot the tetrahedron on the figure
% optional input:
% plab : points label
% flab : faces label
% vlab : volume label (can be element no. e.g.)
% shrinkage : ratio to shrinke the element for better visualisation

if exist('shrinkage', 'var')
    vcentral = sum(x,1)/4;
    for idim = 1:3
        x(:,idim) = vcentral(idim) + (x(:,idim) - vcentral(idim))*shrinkage;
    end
    if exist('x_node', 'var')
        for idim = 1:3
            x_node(:,idim) = vcentral(idim) + (x_node(:,idim) - vcentral(idim))*shrinkage;
        end
        plot3(ax, x_node(:,1), x_node(:,2), x_node(:,3), 'x')
    end
end
if any(ne(size(x), [4,3]))
    error('Input corner coordinates is not in the correct shape [4, 3].');
end
edges = [1,2; 1,3; 1,4; 2,3; 2,4; 3,4];
for e = 1:size(edges, 1)
    edge = edges(e,:);
    plot3(ax, x(edge,1), x(edge,2), x(edge,3), 'k-', LineWidth=2);
end


if exist('plab', 'var')
    for p = 1:4
        text(x(:,1), x(:,2), x(:,3), plab, 'Color','red','FontSize',10);
    end
end
if exist('flab', 'var')
    fcenter = zeros(4,3);
    % face 1: 2-3-4
    fcenter(1,:) = sum(x([2,3,4],:),1)/3;
    % face 2: 1-3-4
    fcenter(2,:) = sum(x([1,3,4],:),1)/3;
    % face 3: 1-2-4
    fcenter(3,:) = sum(x([1,2,4],:),1)/3;
    % face 4: 1-2-3
    fcenter(4,:) = sum(x([1,2,3],:),1)/3;
    for f = 1:4
        text(fcenter(:,1), fcenter(:,2), fcenter(:,3), ...
            flab, 'Color','Green', 'FontSize',10);
    end
end
if exist('vlab', 'var')
    vcenter = sum(x,1)/4;
    text(vcenter(:,1), vcenter(:,2), vcenter(:,3), vlab, ...
        'Color','blue', 'FontSize',10)
end

end