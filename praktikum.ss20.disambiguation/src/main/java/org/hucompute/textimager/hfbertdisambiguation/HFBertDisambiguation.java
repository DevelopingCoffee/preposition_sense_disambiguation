package org.hucompute.textimager.hfbertdisambiguation;

import org.apache.commons.io.FileUtils;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.hucompute.textimager.uima.base.JepAnnotator;
import org.springframework.util.FileSystemUtils;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

public class HFBertDisambiguation extends JepAnnotator {

    private final String[] resourceFiles = new String[] { "python/test.py" };
    private Path tempFolder;

    @Override
    public void initialize(UimaContext aContext) throws ResourceInitializationException {
        super.initialize(aContext);

        if (envName == null || envName.isEmpty()) {
            envName = "textimager_flair_py37";
        }
        if (envPythonVersion == null || envPythonVersion.isEmpty()) {
            envPythonVersion = "3.7";
        }
        if (envDepsConda == null || envDepsConda.isEmpty()) {
            envDepsConda = "uarray=0.6.0 -c conda-forge";
        }
        if (envDepsPip == null || envDepsPip.isEmpty()) {
            envDepsPip = "--upgrade git+https://github.com/flairNLP/flair.git@v0.5.1";
        }
        if (condaVersion == null || condaVersion.isEmpty()) {
            condaVersion = "py37_4.8.3";
        }

        initConda();

        try {

            tempFolder = Files.createTempDirectory(this.getClass().getSimpleName());
            for (String fileName : resourceFiles) {
                Path outPath = Paths.get(tempFolder.toString(), fileName);
                Files.createDirectories(outPath.getParent());
                try (InputStream resourceAsStream = this.getClass().getClassLoader().getResourceAsStream(fileName)) {
                    FileUtils.copyInputStreamToFile(Objects.requireNonNull(resourceAsStream), outPath.toFile());
                }
                catch (Exception e){
                    throw new Exception("resourceAsStream returns null");
                }
            }

            interpreter.exec("import sys, os");
            interpreter.exec("sys.path = ['" + tempFolder.toAbsolutePath().toString() + "/python/'] + sys.path");
            interpreter.exec("from test import Test");
            interpreter.exec("a_test = Test(10)");
            interpreter.exec("a_test.output()");

        } catch (Exception e) {
            throw new ResourceInitializationException(e);
        }
    }

    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {
        // mappingProvider.configure(jCas.getCas());

        System.out.println("Funktioniert process!");

        /*
        Collection<Sentence> sentences = JCasUtil.select(jCas, Sentence.class);
        List<Integer> offsets = sentences.stream().map(Sentence::getBegin).collect(Collectors.toList());
        List<String> sentenceStrings = sentences.stream().map(Sentence::getCoveredText).collect(Collectors.toList());
         */

        try {
            int a = 5;
            /*
            ArrayList<ArrayList<String>> result = (ArrayList<ArrayList<String>>) interpreter.invoke("model.tag",
                    sentenceStrings, offsets);

            for (int i = 0; i < result.size(); i++) {
                ArrayList<String> entry = result.get(i);

                String tagValue = entry.get(0);
                int begin = Integer.parseInt(entry.get(1));
                int end = Integer.parseInt(entry.get(2));

                Type tagType = mappingProvider.getTagType(tagValue);
                AnnotationFS annotation = jCas.getCas().createAnnotation(tagType, begin, end);
                annotation.setStringValue(tagType.getFeatureByBaseName("value"), tagValue);
                jCas.addFsToIndexes(annotation);
            }
             */
        } catch (Exception e) {
            throw new AnalysisEngineProcessException(e);
        }
    }

    @Override
    public void destroy() {
        try {
            if (tempFolder != null) {
                FileSystemUtils.deleteRecursively(tempFolder.toFile());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        super.destroy();
    }
}
