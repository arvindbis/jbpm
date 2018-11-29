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

import java.util.HashMap;
import java.util.Map;

import org.kie.api.task.model.Task;
import org.kie.internal.task.api.prediction.PredictionOutcome;
import org.kie.internal.task.api.prediction.PredictionService;


public class RandomForestPredictionService implements PredictionService {
    
    public static final String IDENTIFIER = "RandomForest";
    
    private double confidenceThreshold = 90.0;
    
    // just for the sake of tests
    private Map<String, Boolean> predictions = new HashMap<>();
    private Map<String, Integer> predictionsConfidence = new HashMap<>();

    public String getIdentifier() {
        return IDENTIFIER;
    }

    public PredictionOutcome predict(Task task, Map<String, Object> inputData) {
        String key = task.getName() + task.getTaskData().getDeploymentId()+ inputData.get("level");
        
        Boolean outcome = predictions.get(key);
        if (outcome == null) {
            return new PredictionOutcome();
        }
        double confidence = predictionsConfidence.get(key).doubleValue();
        
        Map<String, Object> outcomes = new HashMap<>();
        outcomes.put("approved", outcome);
        outcomes.put("confidence", confidence);
        return new PredictionOutcome(confidence, confidenceThreshold, outcomes);
    }

    public void train(Task task, Map<String, Object> inputData, Map<String, Object> outputData) {
        String key = task.getName() + task.getTaskData().getDeploymentId() + inputData.get("level");
        
        predictions.putIfAbsent(key, (Boolean) outputData.get("approved"));
        
        Integer confidenceLevel = predictionsConfidence.getOrDefault(key, 0);
        if (confidenceLevel < 100) {
            predictionsConfidence.put(key, confidenceLevel + 10);
        }
    }

}
