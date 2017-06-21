package regression;

/**
 * @author yasgur99
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.jblas.DoubleMatrix;

public abstract class AbstractRegression {

    protected DoubleMatrix theta; //values of theta that determine cost
    protected DoubleMatrix X; //x input matrix
    protected DoubleMatrix y; //y output vector 
    protected int m; //number of training examples

    /*Gradient Descent*/
    protected double alpha; //the learning rate
    protected double gradientDescentIterations;
    protected List<Double> JHistory;

    /*Feature normalization*/
    protected DoubleMatrix mu; //row vector of the mean of each column from X
    protected DoubleMatrix sigma; //row vector of the range of each column from X

    public AbstractRegression(int n) {
        this.theta = new DoubleMatrix(n, 1);
        this.gradientDescentIterations = 60;
        this.alpha = .1;
        this.JHistory = new ArrayList<>();
    }

    /**
     * Compute the cost of function J(theta) given the values of theta
     *
     * @return the cost given the current theta values
     */
    public abstract double computeCost();

    /**
     * Run gradient descent in order to find the values of theta that minimize
     * cost
     *
     * @return the matrix theta containing their optimized values
     */
    public abstract DoubleMatrix gradientDescent();

    /**
     * Run feature normalization to put the values of X within a smaller range
     * so gradient descent runs faster
     */
    public void featureNormalize() {
        this.mu = X.columnMeans(); //column averages
        this.sigma = X.columnMaxs().sub(X.columnMins()); //range of columns
        this.X = X.subRowVector(mu).divRowVector(sigma);
    }

    /**
     * Read file containing data and loaded into matrix X, vector y, and set m
     *
     * @param fileName the file to read data in from
     * @param loadOnes true if this method should add ones to X for X0
     * @param normalize true if this method should preform feature normalization
     */
    public void load(String fileName, boolean loadOnes, boolean normalize) {
        try {
            //load file
            this.X = DoubleMatrix.loadCSVFile(fileName); //temp
            this.m = this.X.getRows();
            //set vector y and X accordingly
            this.y = this.X.getColumn(X.getColumns() - 1);
            int[] XColumns = new int[X.getColumns() - 1];
            for (int i = 0; i < XColumns.length; i++) {
                XColumns[i] = i;
            }
            this.X = this.X.getColumns(XColumns); // get rid of y from X
            if (normalize) {
                featureNormalize();
            }

            if (loadOnes) {
                this.X = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(m), this.X);
            }

        } catch (IOException ex) {
        }
    }

    public double getResult(int n, double... values) {
        if (this.X.getColumns() != n) {
            throw new IllegalArgumentException("Number of features given does not match this data set.");
        }
        DoubleMatrix normalized = new DoubleMatrix(values).transpose().sub(mu).divRowVector(sigma);
        DoubleMatrix result = DoubleMatrix.concatHorizontally(DoubleMatrix.ones(1), normalized).mmul(theta);
        return result.get(0, 0);
    }

    /**
     * Set the learning rate, alpha
     *
     * @param alpha the learning rate
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Set the number of iterations gradient descent will run for
     *
     * @param iterations the number of iterations
     */
    public void setGradientDescentInterations(double iterations) {
        this.gradientDescentIterations = iterations;
    }
    
    /**
     * Returns the number of iterations preformed in gradient descent
     * @return the number of iterations preformed in gradient descent
     */
     public double getGradientDescentIterations() {
        return gradientDescentIterations;
    }

    /**
     * Returns a matrix containing the current values of theta. After gradient descent
     * being run they will be the optimized values of theta
     *
     * @return the current theta values
     */
    public DoubleMatrix getTheta() {
        return theta;
    }

    /**
     * Returns the X terms of the learning data
     *
     * @return the matrix of X terms of the learning data
     */
    public DoubleMatrix getX() {
        return X;
    }

    /**
     * Returns the Y terms of the learning data
     *
     * @return the matrix of Y terms of the learning data
     */
    public DoubleMatrix getY() {
        return y;
    }

    /**
     * Returns the List of the cost function after each iteration of gradient
     * descent
     *
     * @return list of cost function results after each iteration of gradient
     * descent
     */
    public List<Double> getJHistory() {
        return this.JHistory;
    }

    /**
     * Returns the number of training examples, m
     *
     * @return the number of training examples, m
     */
    public int getM() {
        return m;
    }

    /**
     * Returns a row vector containing the average of each column in X
     * @return a row vector containing the average of each column in X
     */
    public DoubleMatrix getMu() {
        return mu;
    }

    /**
     * Returns a row vector containing the range of each column in X
     * @return a row vector containing the range of each column in X
     */
    public DoubleMatrix getSigma() {
        return sigma;
    }
}
