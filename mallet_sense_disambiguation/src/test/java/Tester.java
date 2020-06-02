import org.junit.Test;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Tester {

    @Test
    public void testRegEx() {
        String test = "b.e5554545 2(1a) On the licensing of overseas editions : A royalty of 6% ( six per cent ) of the Publisher 's net receipts on all copies duplicated <head>by</head> licensed overseas agents .";
        String s = "(.+?)\\s+(.+?)\\s+(.*)";
        Pattern pattern = Pattern.compile(s);
        Matcher matcher = pattern.matcher(test);
        matcher.find();

        System.out.println(matcher.group(1));
        System.out.println(matcher.group(2));
        System.out.println(matcher.group(3));
    }

}
