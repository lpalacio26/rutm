import type { APIRoute } from "astro";
import { sanity } from "../../lib/sanity";
import { urlFor } from "../../lib/sanityImage";

export const GET: APIRoute = async ({ url }) => {
  const offset = Number(url.searchParams.get("offset") ?? 20);
  const limit = 16;

  const [articles, total] = await Promise.all([
    sanity.fetch<any[]>(
      `*[_type=="article"] | order(publishedAt desc)[${offset}...${offset + limit}]{
        title,
        excerpt,
        publishedAt,
        "slug": slug.current,
        heroImage,
        "author": author->{ name, "slug": slug.current },
        "sections": sections[]->{ title, "slug": slug.current },
        "translations": translations[]{language},
        language
      }`
    ),
    sanity.fetch<number>(`count(*[_type=="article"])`),
  ]);

  const items = articles.map((a) => ({
    ...a,
    imageSrc: a.heroImage
      ? urlFor(a.heroImage).width(800).height(520).fit("crop").quality(80).url()
      : null,
  }));

  return new Response(
    JSON.stringify({ articles: items, hasMore: offset + limit < total, nextOffset: offset + limit }),
    { headers: { "Content-Type": "application/json" } }
  );
};
