function testdata(Theta1,Theta2)
X=textread('X_test.txt');
y=textread('y_test.txt');
p=predict(Theta1,Theta2,X);
fprintf('Test data size = %d*%d',size(X,1),size(X,2));
fprintf('\nTest Set Accuracy: %f\n', mean(double(p == y)) * 100);

