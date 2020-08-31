package org.hucompute.textimager.flairdisambiguation;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import jep.JepException;
import org.apache.commons.io.FileUtils;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.dkpro.core.api.parameter.ComponentParameters;
import org.dkpro.core.api.resources.MappingProvider;
import org.hucompute.textimager.uima.base.JepAnnotator;
import org.springframework.util.FileSystemUtils;


public class FlairDisambiguation extends JepAnnotator{

    protected String modelLocation;
    private final String[] resourceFiles = new String[] { "python/test.py" };
    private Path tempFolder;

    public void initialize(UimaContext aContext) throws ResourceInitializationException {
        super.initialize(aContext);

        try {
            tempFolder = Files.createTempDirectory(this.getClass().getSimpleName());
            for (String fileName : resourceFiles) {
                Path outPath = Paths.get(tempFolder.toString(), fileName);
                Files.createDirectories(outPath.getParent());
                try (InputStream resourceAsStream = this.getClass().getClassLoader().getResourceAsStream(fileName)) {
                    FileUtils.copyInputStreamToFile(Objects.requireNonNull(resourceAsStream), outPath.toFile());
                }
            }

            interpreter.exec("from ../"+resourceFiles[0]+" import Test");
            interpreter.exec("test = Test(10)");
            interpreter.exec("test.brrr()");
        } catch (Exception e) {
            throw new ResourceInitializationException(e);
        }
    }

    @Override
    public void process(JCas jCas) {
        try {

            tempFolder = Files.createTempDirectory(this.getClass().getSimpleName());
            for (String fileName : resourceFiles) {
                Path outPath = Paths.get(tempFolder.toString(), fileName);
                Files.createDirectories(outPath.getParent());
                try (InputStream resourceAsStream = this.getClass().getClassLoader().getResourceAsStream(fileName)) {
                    FileUtils.copyInputStreamToFile(Objects.requireNonNull(resourceAsStream), outPath.toFile());
                }
            }

            interpreter.exec("from ../"+resourceFiles[0]+" import Test");
            interpreter.exec("test = Test(10)");
            interpreter.exec("test.brrr()");
        } catch (JepException | IOException e) {
            e.printStackTrace();
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

    public static void main(String...args) {

    }
}
