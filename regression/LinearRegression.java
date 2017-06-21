package regression;

/**
 * @author yasgur99
 */
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class LinearRegression extends AbstractRegression {

    public LinearRegression(int n) {
        super(n);
    }

    /**
     * Compute the cost of function J(theta) given the values of theta
     *
     * @return the cost given the current theta values
     */
    @Override
    public double computeCost() {
        DoubleMatrix hypothesis = X.mmul(theta);
        DoubleMatrix differenceSquared = MatrixFunctions.pow(hypothesis.sub(y), 2);
        return 1.0 / (2.0 * (double) m) * differenceSquared.sum();
    }

    /**
     * Run gradient descent in order to find the values of theta that minimize
     * cost
     *
     * @return the matrix theta containing their optimized values
     */
    @Override
    public DoubleMatrix gradientDescent() {
        for (int i = 0; i < gradientDescentIterations; i++) {
            DoubleMatrix hypothesis = this.X.mmul(theta);
            DoubleMatrix error = hypothesis.sub(y);
            DoubleMatrix changeInTheta = X.transpose().mmul(error).mul(alpha / (double) m);
            this.theta = theta.sub(changeInTheta);
            this.JHistory.add(computeCost()); //save the j history after each iteration
        }
        return theta;
    }
}
