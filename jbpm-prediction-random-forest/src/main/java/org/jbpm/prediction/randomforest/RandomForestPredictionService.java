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
import java.util.HashMap;
import java.util.Map;

import org.kie.api.task.model.Task;
import org.kie.internal.task.api.prediction.PredictionOutcome;
import org.kie.internal.task.api.prediction.PredictionService;
import smile.classification.RandomForest;
import smile.data.Attribute;
import smile.data.AttributeDataset;
import smile.data.NominalAttribute;


public class RandomForestPredictionService implements PredictionService {
    
    public static final String IDENTIFIER = "RandomForest";
    
    private double confidenceThreshold = 90.0;
    private int NUMBER_OF_TREES = 100;

    // Random forest
    private RandomForest randomForest;
    private Attribute approval = new NominalAttribute("user");
    private Attribute approved = new NominalAttribute("approved");
    private AttributeDataset dataset = new AttributeDataset("test", new Attribute[]{approval}, approved);


    public String getIdentifier() {
        return IDENTIFIER;
    }

    /**
     * Converts the normalised Out-Of-Bag error (OOB) into an accuracy measure.
     * @param error An OOB error between 0 (minimum error) and 1 (maximum error).
     * @return An accuracy measure between 0% (minimum accuracy) and 100% (maximum accuracy)
     */
    private static double accuracy(double error) {
        return (1.0 - error) * 100.0;
    }

    public PredictionOutcome predict(Task task, Map<String, Object> inputData) {
        String key = task.getName() + task.getTaskData().getDeploymentId()+ inputData.get("level");


        if (randomForest == null) {
            return new PredictionOutcome();
        } else {
            final Approval _approval = Approval.create((String) inputData.get("ActorId"), (Integer) inputData.get("level"));
            final double[] features;
            try {
                features = new double[]{approval.valueOf(String.valueOf(_approval.hashCode()))};
                final int prediction = randomForest.predict(features);
                System.out.println("Prediction:");
                System.out.println(approved.toString(prediction));
                System.out.println("Error:");
                System.out.println(randomForest.error());
                Map<String, Object> outcomes = new HashMap<>();
                outcomes.put("approved", Boolean.valueOf(approved.toString(prediction)));
                outcomes.put("confidence", accuracy(randomForest.error()));
                return new PredictionOutcome(randomForest.error(), confidenceThreshold, outcomes);
            } catch (ParseException e) {
                e.printStackTrace();
            }
            return new PredictionOutcome();
        }
    }

    public void train(Task task, Map<String, Object> inputData, Map<String, Object> outputData) {
        System.out.println("Training the RF!");
        System.out.println("with:" + outputData.toString());

        final Approval _approval = Approval.create((String) inputData.get("ActorId"), (Integer) inputData.get("level"));
        try {
            dataset.add(new double[]{approval.valueOf(String.valueOf(_approval.hashCode()))}, approved.valueOf(outputData.get("approved").toString()));

        } catch (ParseException e) {
            e.printStackTrace();
        }
        int[] ys = new int[dataset.size()];
        for (int i = 0 ; i < dataset.size() ; i++) {
            ys[i] = (int) dataset.get(i).y;
        }
        if (ys.length >= 2) { // we have enough classes to perform a prediction
            randomForest = new RandomForest(dataset.x(), ys, NUMBER_OF_TREES);
        }
    }

}
