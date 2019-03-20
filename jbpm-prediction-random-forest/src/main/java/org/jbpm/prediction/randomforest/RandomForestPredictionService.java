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

import java.text.ParseException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.kie.api.task.model.Task;
import org.kie.internal.task.api.prediction.PredictionOutcome;
import org.kie.internal.task.api.prediction.PredictionService;

import smile.classification.RandomForest;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;
import smile.data.StringAttribute;


public class RandomForestPredictionService implements PredictionService {
    
    public static final String IDENTIFIER = "RandomForest";
    
    private double confidenceThreshold = 100.0;
    private int NUMBER_OF_TREES = 100;
    private int MIN_COUNT = 5;
    private int count = 0;

    // Random forest
    private RandomForest randomForest;
    private Attribute userName = new StringAttribute("user");
    private Attribute item = new StringAttribute("item");
    private Attribute approved = new NominalAttribute("approved");
    private AttributeDataset dataset = new AttributeDataset("test", new Attribute[]{userName, item}, approved);


    public String getIdentifier() {
        return IDENTIFIER;
    }

    /**
     * Converts the normalised Out-Of-Bag error (OOB) into an accuracy measure.
     * @param error An OOB error between 0 (minimum error) and 1 (maximum error).
     * @return An accuracy measure between 0% (minimum accuracy) and 100% (maximum accuracy)
     */
    private double accuracy(double error) {
    	if (count < MIN_COUNT) {
    		return 0D;
    	}
        return (1.0 - error) * 100.0;
    }

    public PredictionOutcome predict(Task task, Map<String, Object> inputData) {
        if (randomForest == null || !"ManagerApproval".equals(task.getFormName())) {
            return new PredictionOutcome();
        } else {
            try {
            	String itemInput = (String) inputData.get("item");
	        	if (itemInput == null) {
	        		itemInput = "apple";
	        	}
	        	String requestor = (String) inputData.get("requestor");
	        	if (requestor == null) {
	        		requestor = (String) inputData.get("ActorId");
	        	}
            	double[] features = new double[]{
                    userName.valueOf(requestor),
                    item.valueOf(itemInput)
        		};
            	final int prediction = randomForest.predict(features);
                Map<String, Object> outcomes = new HashMap<>();
                double accuracy = accuracy(randomForest.error());
                outcomes.put("approved", Boolean.valueOf(approved.toString(prediction)));
                outcomes.put("confidence", accuracy);
                System.out.print("Predict Input: userName = " + requestor + ", item = " + itemInput);
                System.out.println("; predicting '" + outcomes.get("approved") + "' with accuracy " + accuracy + "%");
                return new PredictionOutcome(accuracy, confidenceThreshold, count, MIN_COUNT, outcomes);
            } catch (ParseException e) {
                e.printStackTrace();
            }
            return new PredictionOutcome();
        }
    }

    public void train(Task task, Map<String, Object> inputData, Map<String, Object> outputData) {
        if ("ManagerApproval".equals(task.getFormName())) {
    		count++;
	        try {
	        	String itemInput = (String) inputData.get("item");
	        	if (itemInput == null) {
	        		itemInput = "apple";
	        	}
	        	String requestor = (String) inputData.get("requestor");
	        	if (requestor == null) {
	        		requestor = (String) inputData.get("ActorId");
	        	}
//		    	System.out.println("Train task " + requestor + " " + itemInput + " = " + outputData.get("approved"));
	            dataset.add(new double[]{
	        		userName.valueOf(requestor),
	                item.valueOf(itemInput)
	            }, approved.valueOf(outputData.get("approved").toString()));
//		    	System.out.println("Train task " + userName.valueOf(requestor) + " " 
//		    			+ item.valueOf(itemInput) + " = " 
//		    			+ approved.valueOf(outputData.get("approved").toString()));
		        int[] ys = new int[dataset.size()];
		        for (int i = 0 ; i < dataset.size() ; i++) {
		            ys[i] = (int) dataset.get(i).y;
		        }
		        
		        if (ys.length >= 2) { // we have enough classes to perform a prediction
		            randomForest = new RandomForest(dataset.x(), ys, NUMBER_OF_TREES);
		        }
	        } catch (Throwable t) {
	            t.printStackTrace();
	        }
    	}
    }

}
