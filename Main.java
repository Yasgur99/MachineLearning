/**
 * @author yasgur99
 */

import regression.LinearRegression;

public class Main {

    public static void main(String[] args) {
        LinearRegression lr = new LinearRegression(3);
        lr.load("/home/yasgur99/Documents/Programming/MachineLearning/res/ex1data2.txt", true, true);
        lr.gradientDescent();
        System.out.println("Optimized values of theta: " + lr.getTheta());
        System.out.println("Result of 1650 and 3: " + lr.getResult(3, 1650, 3));
    }

}
