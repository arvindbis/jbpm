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
import static org.junit.Assert.assertNull;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jbpm.services.api.model.DeploymentUnit;
import org.jbpm.services.api.model.ProcessInstanceDesc;
import org.jbpm.test.services.AbstractKieServicesTest;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.kie.api.runtime.process.ProcessInstance;
import org.kie.api.runtime.query.QueryContext;
import org.kie.api.task.model.TaskSummary;
import org.kie.internal.query.QueryFilter;

public class RandomForestPredictionServiceProcessTest extends AbstractKieServicesTest {

    private List<Long> instances = new ArrayList<>();
    
    @BeforeClass
    public static void setupOnce() {
        System.setProperty("org.jbpm.task.prediction.service", RandomForestPredictionService.IDENTIFIER);
    }
    
    @AfterClass
    public static void cleanOnce() {
        System.clearProperty("org.jbpm.task.prediction.service");
    }
    
    @After
    public void abortInstances() {
        //instances.forEach(processInstanceId -> processService.abortProcessInstance(processInstanceId));
    }
    
    @Override
    protected List<String> getProcessDefinitionFiles() {
        List<String> processes = new ArrayList<String>();
        processes.add("BPMN2-UserTask.bpmn2");
        return processes;
    }

    @Override
    public DeploymentUnit prepareDeploymentUnit() throws Exception {
        // specify GROUP_ID, ARTIFACT_ID, VERSION of your kjar
        return createAndDeployUnit("org.jbpm.test.prediction", "random-forest-test", "1.0.0");
    }
    
    
    @Test
    public void testRandomForestPredictionService() {
        
        Map<String, Object> outputs = startAndReturnTaskOutputData(5, true);
        // first task has no predictions at all
        assertEquals(0, outputs.size());
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(10.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(20.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(30.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(40.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(50.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(60.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(70.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(80.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertEquals(true, outputs.get("approved"));
        assertEquals(90.0, outputs.get("confidence"));
        
        outputs = startAndReturnTaskOutputData(5, true);
        assertNull(outputs);
        
        // make sure all process instances are completed
        Collection<ProcessInstanceDesc> activeInstances = runtimeDataService.getProcessInstances(Collections.singletonList(ProcessInstance.STATE_ACTIVE), null, new QueryContext());
        assertNotNull(activeInstances);
        assertEquals(0, activeInstances.size());
    }

    
    /*
     * Helper methods
     */
    
    protected Map<String, Object> startAndReturnTaskOutputData(Integer level, Boolean approved) {
        Map<String, Object> parameters = new HashMap<>();
        parameters.put("item", "test item");
        parameters.put("level", level);
        long processInstanceId = processService.startProcess(deploymentUnit.getIdentifier(), "UserTask", parameters);
        instances.add(processInstanceId);
        
        List<TaskSummary> tasks = runtimeDataService.getTasksByStatusByProcessInstanceId(processInstanceId, null, new QueryFilter());
        assertNotNull(tasks);
        
        if (!tasks.isEmpty()) {
        
            Long taskId = tasks.get(0).getId();
            
            Map<String, Object> outputs = userTaskService.getTaskOutputContentByTaskId(taskId);
            assertNotNull(outputs);
            
            userTaskService.completeAutoProgress(taskId, "john", Collections.singletonMap("approved", approved));
            
            return outputs;
        }
        
        return null;
    }
}
