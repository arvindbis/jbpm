/*
 * Copyright 2018 Red Hat, Inc. and/or its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.jbpm.prediction.randomforest;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.jbpm.services.api.model.DeploymentUnit;
import org.jbpm.test.services.AbstractKieServicesTest;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.kie.api.task.model.TaskSummary;
import org.kie.internal.query.QueryFilter;

import smile.classification.RandomForest;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.math.Math;
import smile.validation.LOOCV;

public class RandomForestTest {

    @Test
    public void testSmileApproval() {
        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(2);
        try {
        	
            AttributeDataset approval = arffParser.parse(
        		this.getClass().getClassLoader().getResource("approval.nominal.arff").getPath());
            double[][] x = approval.toArray(new double[approval.size()][]);
            int[] y = approval.toArray(new int[approval.size()]);

            int n = x.length;
            LOOCV loocv = new LOOCV(n);
            int error = 0;
            for (int i = 0; i < n; i++) {
                double[][] trainx = Math.slice(x, loocv.train[i]);
                int[] trainy = Math.slice(y, loocv.train[i]);
                
                RandomForest forest = new RandomForest(approval.attributes(), trainx, trainy, 100);
                System.out.println("Error = " + forest.error());
                int prediction = forest.predict(x[loocv.test[i]]);
                if (y[loocv.test[i]] != prediction) {
                    error++;
//                	System.out.println("Incorrectly predicted " + Arrays.toString(x[loocv.test[i]]));
                } else {
//                	System.out.println("Correctly predicted " + Arrays.toString(x[loocv.test[i]]));
                }
            }
            
            System.out.println("Random Forest error = " + error + " out of " + n);
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail();
        }
    }

    @Test
    public void testSmileRecommendationLegal() {
    	testSmileRecommendation(4);
    }
    
    @Test
    public void testSmileRecommendationHR() {
    	testSmileRecommendation(3);
    }
    
    public void testSmileRecommendation(int responseIndex) {
        ArffParser arffParser = new ArffParser();
        arffParser.setResponseIndex(responseIndex);
        try {
        	
            AttributeDataset approval = arffParser.parse(
        		this.getClass().getClassLoader().getResource("recommendation.nominal.arff").getPath());
            double[][] x = approval.toArray(new double[approval.size()][]);
            int[] y = approval.toArray(new int[approval.size()]);

            int n = x.length;
            LOOCV loocv = new LOOCV(n);
            int error = 0;
            RandomForest forest = null;
            for (int i = 0; i < n; i++) {
                double[][] trainx = Math.slice(x, loocv.train[i]);
                int[] trainy = Math.slice(y, loocv.train[i]);
                
                forest = new RandomForest(approval.attributes(), trainx, trainy, 100);
                System.out.println("Error = " + forest.error());
                int prediction = forest.predict(x[loocv.test[i]]);
                if (y[loocv.test[i]] != prediction) {
                    error++;
                } else {
                }
            }
            
            System.out.println("Random Forest error = " + error + " out of " + n);
            System.out.println(forest.getTrees()[0].dot());
            System.out.println(forest.getTrees()[1].dot());
            System.out.println(forest.getTrees()[2].dot());
            System.out.println(forest.getTrees()[3].dot());
            System.out.println(forest.getTrees()[4].dot());
        } catch (Exception ex) {
            ex.printStackTrace();
            Assert.fail();
        }
    }

}
