package org.hucompute.textimager.flairdisambiguation;

import java.io.IOException;
import java.nio.file.Paths;
import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.junit.BeforeClass;
import org.junit.Test;

import com.google.common.io.Files;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;

import org.hucompute.textimager.flairdisambiguation.FlairDisambiguation;

public class TestFlairDisambiguation {

    @BeforeClass
    public static void setUpClass() throws ResourceInitializationException {
    }

    @Test
    public void test() throws UIMAException, IOException {
        FlairDisambiguation flairDisambiguation = new FlairDisambiguation();

        String text = "hallo";
        JCas jCas = JCasFactory.createText(text);

        flairDisambiguation.process(jCas);

        // String home = System.getenv("HOME");
        /* String model_location = home + "/.textimager/models/PharmaCoNER-PCSE_mean-BPEmb-TF-w2v.pt";
        if (!Paths.get(model_location).toFile().exists()) {
            Files.copy(Paths.get(
                    "/resources/public/stoeckel/projects/EsPharmaNER-REST/models/PharmaCoNER-PCSE_mean-BPEmb-TF-w2v.pt")
                    .toFile(), Paths.get(model_location).toFile());
        }
        */

        /*
        JCasUtil.select(jCas, NamedEntity.class).forEach(ner -> {
            System.out.println(ner.getCoveredText() + ": " + ner);
        });
        assert JCasUtil.select(jCas, NamedEntity.class).size() > 0;
         */

    }
}
